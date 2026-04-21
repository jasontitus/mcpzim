// SPDX-License-Identifier: MIT
//
// Minimal Obj-C bridge that converts an NSException raised inside a
// Swift closure into a return value Swift can handle. Used to guard
// AVAudioEngine `installTapOnBus:` / `start:` calls — Foundation APIs
// throw NSExceptions that the Swift runtime can't catch natively, so
// a raised exception kills the whole app. Wrapping the call in
// @try/@catch lets us degrade gracefully to a toast instead of a
// crash.

#import <Foundation/Foundation.h>
#import <AVFAudio/AVFAudio.h>

NS_ASSUME_NONNULL_BEGIN

@interface ObjCExceptionWrapper : NSObject

/// Run `block`. If it raises an NSException, catch it and return its
/// `reason` (the human-readable message). If no exception, return
/// `nil`.
///
/// IMPORTANT: this ONLY catches exceptions raised by the Obj-C code
/// inside `block` itself. If `block` is a Swift closure that calls
/// into AVFAudio / other Obj-C APIs and THOSE raise an NSException,
/// the exception unwinds through the Swift frame first — Swift's
/// exception personality aborts before control returns to the
/// `@try` here. For that case use the specialised installTap
/// helper below, which keeps the raising call inside Obj-C.
+ (nullable NSString *)tryBlock:(void (^NS_NOESCAPE)(void))block;

/// `installTapOnBus:` wrapped in Obj-C @try/@catch. The exception
/// from AVFAudio is raised synchronously from inside `installTap`,
/// so wrapping the AVFAudio call (not a Swift closure) is what
/// actually lets us catch. The tap `block` is still Swift, but
/// that's fine — Core Audio invokes it asynchronously on its own
/// thread after installTap returns, so no exception can unwind
/// through the Swift block during this call.
+ (nullable NSString *)installTapOnNode:(AVAudioNode *)node
                                    bus:(AVAudioNodeBus)bus
                             bufferSize:(AVAudioFrameCount)bufferSize
                                 format:(AVAudioFormat *)format
                                  block:(AVAudioNodeTapBlock)block;

@end

NS_ASSUME_NONNULL_END
