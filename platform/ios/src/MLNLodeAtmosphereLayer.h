// Lode: the space around the globe. A custom layer drawn last on every globe map: deep space, a soft blue glow that
// fades out from the edge, and still stars, like Apple's globe. The Earth itself is left alone. It moves with the
// globe in the same frame (the engine's own projection), so it never lags a pinch. Off on the flat map.
#import "MLNCustomStyleLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface MLNLodeAtmosphereLayer : MLNCustomStyleLayer
@end

NS_ASSUME_NONNULL_END
