// SPDX-License-Identifier: MIT
//
// Objective-C surface for the libzim Obj-C++ bridge. The whole file is a
// no-op unless `<zim/archive.h>` is available — normally because you've
// vendored CoreKiwix.xcframework into MCPZimChat/Frameworks/ and added it
// to the target. Matching guards in `LibzimBridge.mm` and `LibzimReader.swift`
// (the `canImport(CoreKiwix)` branch) keep the scaffolding buildable until
// you're ready to wire up the xcframework.

#import "../Voice/ObjCExceptionWrapper.h"

#if __has_include(<zim/archive.h>)

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ZimEntryBridge : NSObject
@property (readonly, nonatomic, copy) NSString *path;
@property (readonly, nonatomic, copy) NSString *title;
@property (readonly, nonatomic, copy) NSString *mimetype;
@property (readonly, nonatomic, copy) NSData   *content;
@end

@interface ZimSearchHitBridge : NSObject
@property (readonly, nonatomic, copy) NSString *path;
@property (readonly, nonatomic, copy) NSString *title;
@end

/// Thin Obj-C++ wrapper around `zim::Archive`. All methods swallow the
/// underlying C++ exceptions and surface "no result" as `nil` / empty arrays,
/// so Swift callers don't need to understand libzim's error taxonomy.
@interface ZimArchive : NSObject

- (nullable instancetype)initWithFileURL:(NSURL *)url
                                   error:(NSError * _Nullable * _Nullable)outError;

- (BOOL)hasEntry:(NSString *)path;
- (nullable NSString *)metadataValue:(NSString *)key;
- (nullable ZimEntryBridge *)readEntryAtPath:(NSString *)path;
- (nullable ZimEntryBridge *)readMainPage;
- (NSArray<ZimSearchHitBridge *> *)searchFulltext:(NSString *)query
                                            limit:(int32_t)limit;
- (NSArray<ZimSearchHitBridge *> *)suggestTitles:(NSString *)query
                                           limit:(int32_t)limit;

@property (readonly) BOOL hasFulltextIndex;
@property (readonly) BOOL hasTitleIndex;
@property (readonly) int32_t articleCount;

/// Cap this archive's cluster cache (libzim's LRU of decompressed
/// zstd clusters) at `sizeInBytes`. Clusters larger than the cap get
/// dropped immediately after the Blob wrapping their content is
/// released — so a one-shot read of a 700 MB streetzim `graph.bin`
/// doesn't leave 700 MB sitting in libzim's cache forever.
- (void)setClusterCacheMaxSizeBytes:(NSUInteger)sizeInBytes;
@property (readonly) NSUInteger clusterCacheCurrentSizeBytes;
@property (readonly) NSUInteger clusterCacheMaxSizeBytes;

@end

NS_ASSUME_NONNULL_END

#endif
