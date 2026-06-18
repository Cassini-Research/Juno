#import "JunoExceptionGuard.h"

NSError *_Nullable JunoCatchNSException(__attribute__((noescape)) void (^block)(void)) {
    @try {
        block();
        return nil;
    } @catch (NSException *exception) {
        NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
        if (exception.reason) {
            userInfo[NSLocalizedDescriptionKey] = exception.reason;
        }
        if (exception.name) {
            userInfo[@"JunoExceptionName"] = exception.name;
        }
        if (exception.userInfo) {
            userInfo[@"JunoExceptionUserInfo"] = exception.userInfo;
        }
        return [NSError errorWithDomain:@"JunoObjCException" code:1 userInfo:userInfo];
    }
}
