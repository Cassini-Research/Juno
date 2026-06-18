#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs `block`, converting any raised Objective-C `NSException` into a returned
/// `NSError`. Returns `nil` when the block completes normally.
///
/// Swift's `do`/`catch` only intercepts Swift `Error`s — it cannot catch
/// `NSException`s thrown from Objective-C/C++ framework code (e.g. AVFoundation's
/// `-[AVAudioNode installTapOnBus:bufferSize:format:block:]`, which raises when
/// the input node reports an invalid hardware format). Such an exception
/// propagates straight through Swift frames to `std::terminate()`, aborting the
/// process. Wrapping the risky call in this guard lets the Swift caller recover
/// on the normal error path instead of crashing.
NSError *_Nullable JunoCatchNSException(__attribute__((noescape)) void (^block)(void));

NS_ASSUME_NONNULL_END
