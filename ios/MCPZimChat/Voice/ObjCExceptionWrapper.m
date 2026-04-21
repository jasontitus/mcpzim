// SPDX-License-Identifier: MIT

#import "ObjCExceptionWrapper.h"

@implementation ObjCExceptionWrapper

+ (nullable NSString *)tryBlock:(void (^NS_NOESCAPE)(void))block {
    @try {
        block();
        return nil;
    }
    @catch (NSException *e) {
        NSString *reason = e.reason ?: e.name ?: @"unknown";
        return reason;
    }
}

+ (nullable NSString *)installTapOnNode:(AVAudioNode *)node
                                    bus:(AVAudioNodeBus)bus
                             bufferSize:(AVAudioFrameCount)bufferSize
                                 format:(AVAudioFormat *)format
                                  block:(AVAudioNodeTapBlock)block {
    @try {
        [node installTapOnBus:bus bufferSize:bufferSize format:format block:block];
        return nil;
    }
    @catch (NSException *e) {
        return e.reason ?: e.name ?: @"unknown";
    }
}

@end
