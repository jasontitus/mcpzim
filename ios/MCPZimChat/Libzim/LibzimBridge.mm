// SPDX-License-Identifier: MIT
//
// Obj-C++ implementation of the libzim bridge declared in `LibzimBridge.h`.
// Everything here is wrapped in `__has_include(<zim/archive.h>)` so the file
// compiles to empty when CoreKiwix.xcframework isn't linked — meaning you
// can keep these two files in the repo indefinitely and have the rest of
// the app build cleanly before you've wired up the xcframework.

#if __has_include(<zim/archive.h>)

#import "LibzimBridge.h"

#include <zim/archive.h>
#include <zim/entry.h>
#include <zim/error.h>
#include <zim/item.h>
#include <zim/search.h>
#include <zim/suggestion.h>
#include <memory>
#include <string>

namespace {
inline NSString *ToNS(const std::string &s) {
    return [[NSString alloc] initWithBytes:s.data()
                                    length:s.size()
                                  encoding:NSUTF8StringEncoding] ?: @"";
}
}

#pragma mark - ZimEntryBridge

@interface ZimEntryBridge ()
- (instancetype)initWithPath:(NSString *)path
                       title:(NSString *)title
                    mimetype:(NSString *)mimetype
                     content:(NSData *)content;
@end

@implementation ZimEntryBridge
- (instancetype)initWithPath:(NSString *)path
                       title:(NSString *)title
                    mimetype:(NSString *)mimetype
                     content:(NSData *)content {
    if ((self = [super init])) {
        _path = [path copy];
        _title = [title copy];
        _mimetype = [mimetype copy];
        _content = [content copy];
    }
    return self;
}
@end

#pragma mark - ZimSearchHitBridge

@interface ZimSearchHitBridge ()
- (instancetype)initWithPath:(NSString *)path title:(NSString *)title;
@end

@implementation ZimSearchHitBridge
- (instancetype)initWithPath:(NSString *)path title:(NSString *)title {
    if ((self = [super init])) {
        _path = [path copy];
        _title = [title copy];
    }
    return self;
}
@end

#pragma mark - ZimArchive

@implementation ZimArchive {
    std::unique_ptr<zim::Archive> _archive;
}

- (instancetype)initWithFileURL:(NSURL *)url
                          error:(NSError * _Nullable *)outError {
    if ((self = [super init]) == nil) return nil;
    try {
        const char *fs = url.fileSystemRepresentation;
        if (fs == nullptr) {
            if (outError) {
                *outError = [NSError errorWithDomain:@"ZimArchive"
                                                code:EINVAL
                                            userInfo:@{NSLocalizedDescriptionKey: @"URL has no file-system representation"}];
            }
            return nil;
        }
        _archive = std::make_unique<zim::Archive>(std::string(fs));
    } catch (const std::exception &e) {
        if (outError) {
            const char *what = e.what();
            NSString *msg = what ? [NSString stringWithUTF8String:what] : @"libzim error";
            *outError = [NSError errorWithDomain:@"ZimArchive"
                                            code:1
                                        userInfo:@{NSLocalizedDescriptionKey: msg ?: @"libzim error"}];
        }
        return nil;
    } catch (...) {
        if (outError) {
            *outError = [NSError errorWithDomain:@"ZimArchive"
                                            code:1
                                        userInfo:@{NSLocalizedDescriptionKey: @"unknown libzim error"}];
        }
        return nil;
    }
    return self;
}

- (BOOL)hasEntry:(NSString *)path {
    if (!_archive) return NO;
    try {
        return _archive->hasEntryByPath(std::string(path.UTF8String));
    } catch (...) {
        return NO;
    }
}

- (NSString *)metadataValue:(NSString *)key {
    if (!_archive) return nil;
    // libzim has no `hasMetadata`; the canonical check is to try getMetadata
    // and catch EntryNotFound. getMetadataKeys() would also work but allocates
    // the full vector just to membership-test a single key.
    try {
        return ToNS(_archive->getMetadata(std::string(key.UTF8String)));
    } catch (const zim::EntryNotFound &) {
        return nil;
    } catch (...) {
        return nil;
    }
}

- (ZimEntryBridge *)readEntryAtPath:(NSString *)path {
    if (!_archive) return nil;
    try {
        std::string p(path.UTF8String);
        if (!_archive->hasEntryByPath(p)) return nil;
        auto entry = _archive->getEntryByPath(p);
        auto item = entry.getItem(/*followRedirect=*/true);
        auto blob = item.getData();
        NSData *data = [NSData dataWithBytes:blob.data() length:(NSUInteger)blob.size()];
        return [[ZimEntryBridge alloc] initWithPath:ToNS(entry.getPath())
                                              title:ToNS(entry.getTitle())
                                           mimetype:ToNS(item.getMimetype())
                                            content:data];
    } catch (...) {
        return nil;
    }
}

- (ZimEntryBridge *)readMainPage {
    if (!_archive) return nil;
    try {
        if (!_archive->hasMainEntry()) return nil;
        auto entry = _archive->getMainEntry();
        auto item = entry.getItem(/*followRedirect=*/true);
        auto blob = item.getData();
        NSData *data = [NSData dataWithBytes:blob.data() length:(NSUInteger)blob.size()];
        // Use the ITEM's path (post-redirect target, e.g. "index.html"),
        // not the entry's (pre-redirect, e.g. "mainPage") — otherwise a
        // ZIM whose main entry is a redirect yields an unusable path
        // that won't round-trip through `read(path:)`.
        return [[ZimEntryBridge alloc] initWithPath:ToNS(item.getPath())
                                              title:ToNS(entry.getTitle())
                                           mimetype:ToNS(item.getMimetype())
                                            content:data];
    } catch (...) {
        return nil;
    }
}

- (NSArray<ZimSearchHitBridge *> *)searchFulltext:(NSString *)query limit:(int32_t)limit {
    if (!_archive) return @[];
    if (limit <= 0) return @[];
    try {
        if (!_archive->hasFulltextIndex()) return @[];
        zim::Searcher searcher(*_archive);
        zim::Query q;
        q.setQuery(std::string(query.UTF8String));
        auto search = searcher.search(q);
        auto results = search.getResults(0, (int)limit);
        NSMutableArray<ZimSearchHitBridge *> *out = [NSMutableArray arrayWithCapacity:(NSUInteger)limit];
        for (auto it = results.begin(); it != results.end(); ++it) {
            auto entry = *it;
            [out addObject:[[ZimSearchHitBridge alloc] initWithPath:ToNS(entry.getPath())
                                                              title:ToNS(entry.getTitle())]];
        }
        return out;
    } catch (...) {
        return @[];
    }
}

- (NSArray<ZimSearchHitBridge *> *)suggestTitles:(NSString *)query limit:(int32_t)limit {
    if (!_archive) return @[];
    if (limit <= 0) return @[];
    try {
        if (!_archive->hasTitleIndex()) return @[];
        zim::SuggestionSearcher searcher(*_archive);
        auto search = searcher.suggest(std::string(query.UTF8String));
        auto results = search.getResults(0, (int)limit);
        NSMutableArray<ZimSearchHitBridge *> *out = [NSMutableArray arrayWithCapacity:(NSUInteger)limit];
        for (auto it = results.begin(); it != results.end(); ++it) {
            auto hit = *it;
            [out addObject:[[ZimSearchHitBridge alloc] initWithPath:ToNS(hit.getPath())
                                                              title:ToNS(hit.getTitle())]];
        }
        return out;
    } catch (...) {
        return @[];
    }
}

- (BOOL)hasFulltextIndex {
    if (!_archive) return NO;
    try { return _archive->hasFulltextIndex(); } catch (...) { return NO; }
}

- (BOOL)hasTitleIndex {
    if (!_archive) return NO;
    try { return _archive->hasTitleIndex(); } catch (...) { return NO; }
}

- (int32_t)articleCount {
    if (!_archive) return 0;
    try { return (int32_t)_archive->getEntryCount(); } catch (...) { return 0; }
}

@end

#endif // __has_include(<zim/archive.h>)
