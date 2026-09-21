"""Exact full-vocabulary KL, bounded across both token and vocabulary axes.

Uses native reference weights and optional Q1 candidate head; never loads all
logits, reference probabilities, or head deltas simultaneously. No top-k proxy.
"""
import torch


def mixed_head(reference, candidate, probability, start, end):
    ref = reference[start:end].float()
    if candidate is None:
        return ref
    return ref + probability.float() * (candidate.rows(start, end, reference.device, torch.float32) - ref)


class FullVocabularyKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, reference_head, probability, candidate, token_chunk, vocab_chunk):
        if probability.ndim != 0 or any(t.device != student.device for t in (teacher, reference_head, probability)):
            raise ValueError('Scalar probability and matching devices required')
        if student.shape != teacher.shape or student.ndim != 2 or not student.shape[0]:
            raise ValueError('Matching nonempty [tokens,hidden] states required')
        if reference_head.requires_grad or teacher.requires_grad:
            raise ValueError('Reference model must be frozen')
        if token_chunk <= 0 or vocab_chunk <= 0:
            raise ValueError('Positive chunk sizes required')
        norms = []
        total = torch.zeros((), device=student.device, dtype=torch.float32)
        for t in range(0, len(student), token_chunk):
            s, r = student[t:t+token_chunk].float(), teacher[t:t+token_chunk].float()
            zs = torch.full((len(s),), -torch.inf, device=s.device)
            zr = torch.full_like(zs, -torch.inf)
            for v in range(0, len(reference_head), vocab_chunk):
                end = min(v+vocab_chunk, len(reference_head))
                ls = s @ mixed_head(reference_head, candidate, probability, v, end).T
                lr = r @ reference_head[v:end].float().T
                zs = torch.logaddexp(zs, ls.logsumexp(-1))
                zr = torch.logaddexp(zr, lr.logsumexp(-1))
            norms.append(torch.stack((zs, zr)))
            for v in range(0, len(reference_head), vocab_chunk):
                end = min(v+vocab_chunk, len(reference_head))
                ls = s @ mixed_head(reference_head, candidate, probability, v, end).T - zs[:, None]
                lr = r @ reference_head[v:end].float().T - zr[:, None]
                total += (lr.exp() * (lr-ls)).sum()
        ctx.save_for_backward(student, teacher, reference_head, probability, torch.cat(norms, dim=1))
        ctx.candidate, ctx.token_chunk, ctx.vocab_chunk = candidate, token_chunk, vocab_chunk
        return total / len(student)

    @staticmethod
    def backward(ctx, grad):
        student, teacher, reference_head, probability, norms = ctx.saved_tensors
        gs = torch.zeros_like(student, dtype=torch.float32)
        gp = torch.zeros_like(probability, dtype=torch.float32)
        for t in range(0, len(student), ctx.token_chunk):
            s, r = student[t:t+ctx.token_chunk].float(), teacher[t:t+ctx.token_chunk].float()
            zs, zr = norms[:, t:t+len(s)]
            for v in range(0, len(reference_head), ctx.vocab_chunk):
                end = min(v+ctx.vocab_chunk, len(reference_head))
                ref = reference_head[v:end].float()
                mixed = mixed_head(reference_head, ctx.candidate, probability, v, end)
                ps = (s @ mixed.T - zs[:, None]).exp()
                pr = (r @ ref.T - zr[:, None]).exp()
                g = (ps-pr) * (grad.float() / len(student))
                gs[t:t+len(s)].add_(g @ mixed)
                if ctx.candidate is not None:
                    delta = ctx.candidate.rows(v, end, s.device, torch.float32) - ref
                    gp.add_(((g.T @ s) * delta).sum())
        return gs.to(student.dtype), None, None, gp.to(probability.dtype), None, None, None


def full_kl(student, teacher, reference_head, probability, candidate=None, token_chunk=32, vocab_chunk=2048):
    return FullVocabularyKL.apply(student, teacher, reference_head, probability, candidate, token_chunk, vocab_chunk)
