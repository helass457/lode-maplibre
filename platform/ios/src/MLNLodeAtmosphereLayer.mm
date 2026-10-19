#import "MLNLodeAtmosphereLayer.h"
#import "MLNMapView.h"
#import "MLNBackendResource.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

// One full-screen triangle. Each pixel's view ray is taken back through the globe's projection into unit-sphere space:
// a ray that passes within 1 of the center hits the Earth (left alone, but for a faint rim of light at its edge);
// anything else is space, lit by a glow that fades with the distance from the edge, with sparse stars fixed in the sky.
static const char *kLodeAtmosphereShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LodeAtmoUniforms {
  float4x4 inv;     // clip space -> unit sphere space
  float2 size;      // drawable size in pixels
  float alpha;      // how much of it shows (the globe's projection transition)
  float scale;      // pixels per point
  float4 sun;       // xyz: toward the sun, in unit-sphere space (x 90°E, y north, z null island); w: 1 with city lights
  float4 moonC;     // xyz: the view ray through the Moon's center on screen; w: cos of its radius as an angle
  float4 moonSun;   // xyz: the sun as the Moon sees it, turned into that ray's frame (its real phase); w: 1 in front of the Earth
  float4 moonNorth; // xyz: the Moon's north in that frame (its seas sit where they really are); w: how much it shows
  float4 moonMore;  // x: the Moon photo's detail level for its size on screen
};
struct LodeAtmoOut { float4 pos [[position]]; };

vertex LodeAtmoOut lodeAtmoVertex(uint vid [[vertex_id]]) {
  LodeAtmoOut o;
  o.pos = float4(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0, 0.0, 1.0);
  return o;
}

// an integer hash: the float one repeated along columns, so the stars came out in stripes (Oct 17)
static float lodeHash(float2 p) {
  uint2 q = uint2(int2(p) + int2(4096));
  uint n = q.x * 1597334677u ^ q.y * 3812015801u;
  n = (n ^ (n >> 16)) * 2246822519u;
  n = (n ^ (n >> 13)) * 3266489917u;
  n ^= n >> 16;
  return float(n) * (1.0 / 4294967296.0);
}

// The Moon: NASA's photo map of it (Lunar Reconnaissance Orbiter, public domain) on a small sphere, the side that
// faces the Earth toward us, lit by the real sun so its phase is tonight's. Lit the way the Moon is (it stays about
// as bright to the edge, not shaded like a ball: Lommel-Seeliger), a soft terminator, and faint earthshine on the dark
// part. It replaced a plain gray ball with painted-on seas that looked low quality (owner, Oct 18). rgb, a (0 off it).
static float4 lodeMoon(float3 d, constant LodeAtmoUniforms &u, float pixAng, texture2d<float> photo) {
  if (u.moonNorth.w <= 0.0) return float4(0.0);
  float3 c = u.moonC.xyz;
  float cd = dot(d, c);
  if (cd < u.moonC.w - 0.002) return float4(0.0);
  float sinr = sqrt(max(1e-8, 1.0 - u.moonC.w * u.moonC.w));
  float3 v = (d - c * cd) / sinr;
  float len = length(v);
  float edge = smoothstep(1.0, 1.0 - 1.5 * pixAng / sinr, len);
  if (edge <= 0.0) return float4(0.0);
  float3 f = -c; // the side facing us (and the Earth)
  float3 n = normalize(v + f * sqrt(max(0.0, 1.0 - len * len)));
  float3 north = normalize(u.moonNorth.xyz - f * dot(u.moonNorth.xyz, f));
  float3 east = cross(north, f);
  float lon = atan2(dot(n, east), dot(n, f)), lat = asin(clamp(dot(n, north), -1.0, 1.0));
  constexpr sampler msmp(filter::linear, mip_filter::linear, address::repeat);
  float3 albedo = photo.sample(msmp, float2(lon / (2.0 * M_PI_F) + 0.5, 0.5 - lat / M_PI_F), level(u.moonMore.x)).rgb;
  float mu0 = dot(n, u.moonSun.xyz), mu = max(dot(n, f), 0.05);
  float sunlit = clamp(2.0 * max(mu0, 0.0) / (max(mu0, 0.0) + mu), 0.0, 1.25) * smoothstep(-0.02, 0.08, mu0);
  float3 col = albedo * (sunlit * 1.08 + 0.045 * float3(0.8, 0.9, 1.0));
  return float4(min(col, float3(1.0)), edge * u.moonNorth.w);
}

fragment float4 lodeAtmoFragment(LodeAtmoOut in [[stage_in]], constant LodeAtmoUniforms &u [[buffer(0)]],
                                 texture2d<float> lights [[texture(0)]], texture2d<float> moonPhoto [[texture(1)]]) {
  constexpr sampler smp(filter::linear, mip_filter::linear, address::repeat);
  float2 ndc = float2(in.pos.x / u.size.x * 2.0 - 1.0, 1.0 - in.pos.y / u.size.y * 2.0);
  float4 a = u.inv * float4(ndc, -0.5, 1.0);
  float4 b = u.inv * float4(ndc, 0.5, 1.0);
  float3 o = a.xyz / a.w;
  float3 d = normalize(b.xyz / b.w - o);
  // how far apart neighboring pixels' rays point (for the stars below; taken here, where every pixel still runs)
  float pixAng = max(max(length(dfdx(d)), length(dfdy(d))), 1e-5);
  float dist = length(o - dot(o, d) * d); // how close this pixel's ray passes to the Earth's center (radius 1)
  float4 moon = lodeMoon(d, u, pixAng, moonPhoto);
  if (dist < 1.0) {
    // the Moon passing in front of the Earth (behind it, the Earth hides it)
    if (u.moonSun.w > 0.5 && moon.a > 0.0) return float4(moon.rgb, moon.a * u.alpha);
    // on the Earth: the side in darkness right now is shaded, with a soft edge between day and night (where the ray
    // first meets the sphere, its height above the sun's horizon there); a faint rim of light at the edge
    float tHit = -dot(o, d) - sqrt(max(0.0, 1.0 - dist * dist));
    float3 p = o + tHit * d;
    float3 n = normalize(p);
    float sunUp = dot(n, u.sun.xyz);
    float dark = smoothstep(0.08, -0.12, sunUp);
    float night = dark * 0.62;
    // city lights on the night side (NASA's Black Marble 2016, 4096 px): coming on as the sun goes down. They were a
    // flat warm overlay and looked painted on (owner, Oct 18): now dim towns glow sodium orange and bright city cores
    // warm white, with a soft halo around them, like photos from the space station. The detail level is worked out
    // from the pixel's footprint on the sphere (the texture's own mip choice breaks at the date line, where u wraps).
    float lit = 0.0;
    float3 lightCol = float3(1.0, 0.83, 0.55);
    if (u.sun.w > 0.5 && dark > 0.0) {
      float2 uv = float2(atan2(n.x, n.z) / (2.0 * M_PI_F) + 0.5, 0.5 - asin(clamp(n.y, -1.0, 1.0)) / M_PI_F);
      float foot = pixAng * max(tHit, 1e-3) / max(dot(n, -d), 0.25);
      float lod = max(0.0, log2(foot * float(lights.get_width()) / (2.0 * M_PI_F)));
      float v = lights.sample(smp, uv, level(lod)).r;
      float halo = lights.sample(smp, uv, level(lod + 2.5)).r;
      float core = clamp(pow(v, 1.15) * 1.8, 0.0, 1.0);
      float glowL = clamp(halo * 1.5, 0.0, 1.0) * 0.5;
      lit = clamp(core + glowL * (1.0 - core), 0.0, 1.0) * smoothstep(0.0, -0.1, sunUp);
      lightCol = mix(float3(1.0, 0.56, 0.22), float3(1.0, 0.91, 0.76), smoothstep(0.2, 0.85, v));
    }
    float rim = smoothstep(0.95, 1.0, dist) * 0.2 * mix(0.35, 1.0, smoothstep(-0.35, 0.45, sunUp)); // brighter on the sunlit edge
    float a = max(night + lit * (1.0 - night), rim);
    if (a <= 0.003) discard_fragment();
    float3 col = rim > night + lit ? float3(0.6, 0.78, 1.0)
      : (float3(0.0, 0.01, 0.035) * night * (1.0 - lit) + lightCol * lit) / max(a, 0.001);
    return float4(col, a * u.alpha);
  }
  float g = dist - 1.0;
  // the glow follows the sun: full on the sunlit edge of the Earth, about a third on the night edge (the point on
  // the sphere nearest this ray tells which side it's beside)
  float3 limb = normalize(o - dot(o, d) * d);
  float sunSide = mix(0.35, 1.0, smoothstep(-0.35, 0.45, dot(limb, u.sun.xyz)));
  float glow = clamp((exp(-g * 12.0) * 0.5 + exp(-g * 42.0) * 0.38) * sunSide, 0.0, 1.0);
  float3 col = mix(float3(0.010, 0.013, 0.026), float3(0.5, 0.72, 1.0), glow);
  // stars: fixed in the sky around the Earth, so they turn with the globe as you spin it (they were fixed to the
  // screen, which looked painted on, Oct 18). The sky is cut into the six faces of a cube around the Earth, each face
  // into cells of about 4 points; at most one star per cell, at a random spot in it, a small round point (soft edge,
  // varied size and brightness, a few warm or cool), fading out in the glow. The cell size is rounded to a power of
  // two so it doesn't shimmer as the view changes.
  float3 ad = abs(d);
  float2 uv; float face;
  if (ad.x >= ad.y && ad.x >= ad.z) { uv = d.yz / ad.x; face = d.x > 0.0 ? 0.0 : 1.0; }
  else if (ad.y >= ad.z) { uv = d.xz / ad.y; face = d.y > 0.0 ? 2.0 : 3.0; }
  else { uv = d.xy / ad.z; face = d.z > 0.0 ? 4.0 : 5.0; }
  float cellA = exp2(round(log2(4.0 * u.scale * pixAng)));
  float2 cell = floor(uv / cellA);
  float2 key = cell + float2(face * 1000.0, 0.0);
  float h = lodeHash(key);
  if (h > 0.972) {
    float2 spot = (cell + float2(lodeHash(key + 17.0), lodeHash(key + 41.0)) * 0.8 + 0.1) * cellA;
    float k = (h - 0.972) / 0.028;
    float r = (0.35 + 0.55 * k * k) * u.scale;
    float lit = smoothstep(r, r * 0.25, length(uv - spot) / pixAng) * (0.35 + 0.65 * k);
    float tint = lodeHash(key + 7.0);
    float3 star = tint > 0.9 ? float3(1.0, 0.86, 0.7) : tint < 0.1 ? float3(0.75, 0.85, 1.0) : float3(1.0);
    col += star * lit * (1.0 - clamp(glow * 2.2, 0.0, 1.0));
  }
  col = mix(col, moon.rgb, moon.a);
  return float4(col, u.alpha);
}
)METAL";

typedef struct {
  simd_float4x4 inv;
  simd_float2 size;
  float alpha;
  float scale;
  simd_float4 sun;
  simd_float4 moonC;
  simd_float4 moonSun;
  simd_float4 moonNorth;
  simd_float4 moonMore;
} LodeAtmoUniforms;

// toward the sun right now, in unit-sphere space: the point where it's overhead (declination by the day of the year,
// longitude by the time of day; the equation of time, under 17 minutes, is left out)
static simd_float4 LodeSunDirection(void) {
  static simd_float4 cached;
  static NSTimeInterval cachedAt = 0;
  NSDate *now = [NSDate date];
  if (now.timeIntervalSince1970 - cachedAt < 60) return cached; // it moves a quarter of a degree a minute
  cachedAt = now.timeIntervalSince1970;
  NSCalendar *utc = [[NSCalendar alloc] initWithCalendarIdentifier:NSCalendarIdentifierGregorian];
  utc.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];
  const NSInteger day = [utc ordinalityOfUnit:NSCalendarUnitDay inUnit:NSCalendarUnitYear forDate:now];
  NSDateComponents *c = [utc components:NSCalendarUnitHour | NSCalendarUnitMinute | NSCalendarUnitSecond fromDate:now];
  const double hours = c.hour + c.minute / 60.0 + c.second / 3600.0;
  const double decl = -23.44 * M_PI / 180.0 * cos(2.0 * M_PI / 365.0 * (day + 10));
  const double lon = (-15.0 * (hours - 12.0)) * M_PI / 180.0;
  cached = simd_make_float4((float)(sin(lon) * cos(decl)), (float)sin(decl), (float)(cos(lon) * cos(decl)), 0.0f);
  return cached;
}

// toward the Moon right now, in unit-sphere space (the Astronomical Almanac's low-precision formulas, within about
// half a degree): its ecliptic longitude and latitude, then right ascension and declination, then the longitude where
// it's overhead from Greenwich sidereal time
static simd_double3 LodeMoonDirection(void) {
  static simd_double3 cached;
  static NSTimeInterval cachedAt = 0;
  const NSTimeInterval t = [NSDate date].timeIntervalSince1970;
  if (t - cachedAt < 60) return cached;
  cachedAt = t;
  const double r = M_PI / 180.0, d = t / 86400.0 + 2440587.5 - 2451545.0;
  const double L = 218.316 + 13.176396 * d, M = 134.963 + 13.064993 * d, F = 93.272 + 13.229350 * d;
  const double lam = (L + 6.289 * sin(M * r)) * r, bet = 5.128 * sin(F * r) * r, eps = 23.439 * r;
  const double x = cos(bet) * cos(lam), y = cos(eps) * cos(bet) * sin(lam) - sin(eps) * sin(bet);
  const double z = sin(eps) * cos(bet) * sin(lam) + cos(eps) * sin(bet);
  const double ra = atan2(y, x), dec = asin(fmax(-1.0, fmin(1.0, z)));
  const double gmst = fmod(280.46061837 + 360.98564736629 * d, 360.0) * r;
  const double lon = ra - gmst;
  cached = simd_make_double3(sin(lon) * cos(dec), sin(dec), cos(lon) * cos(dec));
  return cached;
}
// turns x by the rotation that takes unit vector a to unit vector b
static simd_double3 LodeTurn(simd_double3 x, simd_double3 a, simd_double3 b) {
  const simd_double3 v = simd_cross(a, b);
  const double c = simd_dot(a, b);
  if (c < -0.9999) { // opposite: half a turn about any axis square to a
    simd_double3 ax = simd_normalize(simd_cross(a, fabs(a.y) < 0.9 ? simd_make_double3(0, 1, 0) : simd_make_double3(1, 0, 0)));
    return 2.0 * simd_dot(ax, x) * ax - x;
  }
  return x * c + simd_cross(v, x) + v * (simd_dot(v, x) / (1.0 + c));
}
// a point in unit-sphere space -> pixels on the drawing surface, and a pixel -> its view ray
static simd_double2 LodePx(simd_double4x4 proj, simd_double3 p, CGSize sz) {
  const simd_double4 c = simd_mul(proj, simd_make_double4(p, 1.0));
  return simd_make_double2((c.x / c.w + 1.0) * 0.5 * sz.width, (1.0 - c.y / c.w) * 0.5 * sz.height);
}
static simd_double3 LodeRay(simd_double4x4 inv, simd_double2 px, CGSize sz) {
  const double nx = px.x / sz.width * 2.0 - 1.0, ny = 1.0 - px.y / sz.height * 2.0;
  const simd_double4 a = simd_mul(inv, simd_make_double4(nx, ny, -0.5, 1.0)), b = simd_mul(inv, simd_make_double4(nx, ny, 0.5, 1.0));
  return simd_normalize(b.xyz / b.w - a.xyz / a.w);
}

@implementation MLNLodeAtmosphereLayer {
  id<MTLRenderPipelineState> _pipeline;
  id<MTLDepthStencilState> _depthStencil;
  id<MTLDevice> _device;
  id<MTLTexture> _lights;      // the city lights, once Lode has downloaded them
  id<MTLTexture> _noLights;    // 1×1 black, bound until then
  BOOL _lightsLoading;
  id<MTLTexture> _moon;        // NASA's Moon map, once Lode has downloaded it (no file, no Moon)
  BOOL _moonLoading;
  NSTimeInterval _moonLookedAt;
  NSTimeInterval _lookedAt;
}

// Lode downloads the city lights to Library/Caches: lode-night-lights-2.jpg (NASA's Black Marble 2016, 4096 × 2048 gray,
// Oct 18) or, before that, lode-night-lights.jpg (2048 × 1024). Made into a one-channel texture with mipmaps (8 MB
// instead of 32 as color, and smooth when the globe shrinks it), off the drawing thread.
- (void)lookForLights {
  if (_lights || _lightsLoading || !_device) return;
  const NSTimeInterval now = [NSDate date].timeIntervalSince1970;
  if (now - _lookedAt < 5) return;
  _lookedAt = now;
  NSString *caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject;
  NSString *file = [caches stringByAppendingPathComponent:@"lode-night-lights-2.jpg"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:file]) file = [caches stringByAppendingPathComponent:@"lode-night-lights.jpg"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:file]) return;
  _lightsLoading = YES;
  id<MTLDevice> device = _device;
  __weak MLNLodeAtmosphereLayer *weakSelf = self;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    id<MTLTexture> tex = nil;
    UIImage *img = [UIImage imageWithContentsOfFile:file];
    CGImageRef cg = img.CGImage;
    if (cg) {
      const size_t w = CGImageGetWidth(cg), h = CGImageGetHeight(cg);
      NSMutableData *buf = [NSMutableData dataWithLength:w * h];
      CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
      CGContextRef ctx = CGBitmapContextCreate(buf.mutableBytes, w, h, 8, w, gray, kCGImageAlphaNone);
      CGColorSpaceRelease(gray);
      if (ctx) {
        CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg);
        CGContextRelease(ctx);
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:w height:h mipmapped:YES];
        td.usage = MTLTextureUsageShaderRead;
        tex = [device newTextureWithDescriptor:td];
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:buf.bytes bytesPerRow:w];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit generateMipmapsForTexture:tex];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
      }
    }
    if (!tex) NSLog(@"Lode atmosphere: no city lights from %@", file);
    dispatch_async(dispatch_get_main_queue(), ^{
      MLNLodeAtmosphereLayer *me = weakSelf;
      if (!me) return;
      me->_lightsLoading = NO;
      if (me->_device == device) me->_lights = tex;
    });
  });
}

// Lode downloads the Moon's map to Library/Caches/lode-moon.jpg (2048 × 1024); the Moon shows only once it's there, so
// the app can take it away again by deleting the file
- (void)lookForMoon {
  if (_moon || _moonLoading || !_device) return;
  const NSTimeInterval now = [NSDate date].timeIntervalSince1970;
  if (now - _moonLookedAt < 5) return;
  _moonLookedAt = now;
  NSString *caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject;
  NSString *file = [caches stringByAppendingPathComponent:@"lode-moon.jpg"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:file]) return;
  _moonLoading = YES;
  id<MTLDevice> device = _device;
  __weak MLNLodeAtmosphereLayer *weakSelf = self;
  MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice:device];
  [loader newTextureWithContentsOfURL:[NSURL fileURLWithPath:file]
                              options:@{ MTKTextureLoaderOptionSRGB : @NO, MTKTextureLoaderOptionGenerateMipmaps : @YES }
                    completionHandler:^(id<MTLTexture> tex, NSError *error) {
    if (!tex) NSLog(@"Lode atmosphere: no Moon: %@", error);
    dispatch_async(dispatch_get_main_queue(), ^{
      MLNLodeAtmosphereLayer *me = weakSelf;
      if (!me) return;
      me->_moonLoading = NO;
      if (me->_device == device) me->_moon = tex;
    });
  }];
}

- (void)didMoveToMapView:(MLNMapView *)mapView {
#if MLN_RENDER_BACKEND_METAL
  MLNBackendResource *resource = [mapView backendResource];
  id<MTLDevice> device = resource.device;
  MTKView *view = resource.mtkView;
  if (!device || !view) return;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kLodeAtmosphereShader] options:nil error:&error];
  if (!library) { NSLog(@"Lode atmosphere: shader didn't compile: %@", error); return; }
  MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
  desc.label = @"Lode atmosphere";
  desc.vertexFunction = [library newFunctionWithName:@"lodeAtmoVertex"];
  desc.fragmentFunction = [library newFunctionWithName:@"lodeAtmoFragment"];
  desc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
  desc.colorAttachments[0].blendingEnabled = YES;
  desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
  desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  desc.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
  desc.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;
  desc.rasterSampleCount = view.sampleCount;
  _pipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!_pipeline) { NSLog(@"Lode atmosphere: no pipeline: %@", error); return; }
  MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
  ds.depthCompareFunction = MTLCompareFunctionAlways;
  ds.depthWriteEnabled = NO;
  _depthStencil = [device newDepthStencilStateWithDescriptor:ds];
  _device = device;
  MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
  _noLights = [device newTextureWithDescriptor:td];
  const uint32_t black = 0xFF000000;
  [_noLights replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&black bytesPerRow:4];
#endif
}

- (void)willMoveFromMapView:(MLNMapView *)mapView {
  _pipeline = nil;
  _depthStencil = nil;
  _lights = nil;
  _moon = nil;
  _noLights = nil;
  _device = nil;
}

- (void)drawInMapView:(MLNMapView *)mapView withContext:(MLNStyleLayerDrawingContext)context {
#if MLN_RENDER_BACKEND_METAL
  // only on the globe, and only while there's space to see (zoomed in, the Earth fills the screen)
  if (!_pipeline || !context.globe || context.projectionTransition <= 0.01 || context.zoomLevel > 6.5) return;
  id<MTLRenderCommandEncoder> encoder = self.renderEncoder;
  if (!encoder) return;
  const MLNMatrix4 m = context.globeProjectionMatrix; // column-major: mXY is column X, row Y
  const simd_double4x4 proj = simd_matrix(simd_make_double4(m.m00, m.m01, m.m02, m.m03), simd_make_double4(m.m10, m.m11, m.m12, m.m13),
                                          simd_make_double4(m.m20, m.m21, m.m22, m.m23), simd_make_double4(m.m30, m.m31, m.m32, m.m33));
  const simd_double4x4 inv = simd_inverse(proj);
  LodeAtmoUniforms u;
  u.inv = simd_matrix(simd_make_float4((float)inv.columns[0].x, (float)inv.columns[0].y, (float)inv.columns[0].z, (float)inv.columns[0].w),
                      simd_make_float4((float)inv.columns[1].x, (float)inv.columns[1].y, (float)inv.columns[1].z, (float)inv.columns[1].w),
                      simd_make_float4((float)inv.columns[2].x, (float)inv.columns[2].y, (float)inv.columns[2].z, (float)inv.columns[2].w),
                      simd_make_float4((float)inv.columns[3].x, (float)inv.columns[3].y, (float)inv.columns[3].z, (float)inv.columns[3].w));
  // the drawing surface in pixels: the map view's own scale is 1 (the Metal view inside it has the screen's 3), and with
  // points here the sphere was worked out a third of its size from the top-left corner, and space was painted over
  // the real globe everywhere else (Oct 17: "the globe goes to the top left")
  CGSize drawable = [mapView backendResource].mtkView.drawableSize;
  if (drawable.width <= 0 || drawable.height <= 0) drawable = CGSizeMake(context.size.width * UIScreen.mainScreen.nativeScale, context.size.height * UIScreen.mainScreen.nativeScale);
  const CGFloat scale = context.size.width > 0 ? drawable.width / context.size.width : 1;
  u.size = simd_make_float2((float)drawable.width, (float)drawable.height);
  u.alpha = (float)fmin(1.0, fmax(0.0, context.projectionTransition));
  u.scale = (float)scale;
  u.sun = LodeSunDirection();
  [self lookForLights];
  u.sun.w = _lights ? 1.0f : 0.0f;
  // The Moon, beside the globe in the direction it really is from here, in its real phase. At its true distance (60
  // Earths away) it would almost never be on screen, so it sits just outside the glow, kept on the screen and clear of
  // the search sheet at the bottom; in front of the Earth or hidden behind it as it really is. Only on the whole
  // globe: it fades out as you zoom in.
  u.moonC = simd_make_float4(0, 0, 0, 1);
  u.moonSun = simd_make_float4(0, 0, 0, 0);
  u.moonNorth = simd_make_float4(0, 0, 0, 0);
  u.moonMore = simd_make_float4(0, 0, 0, 0);
  [self lookForMoon];
  const double moonShow = _moon ? fmin(1.0, fmax(0.0, (2.6 - context.zoomLevel) / 0.6)) * u.alpha : 0.0;
  if (moonShow > 0.01) {
    const CGSize sz = drawable;
    const simd_double3 m = LodeMoonDirection();
    const simd_double2 pc = LodePx(proj, simd_make_double3(0, 0, 0), sz);
    const simd_double3 axis = LodeRay(inv, pc, sz); // from the eye toward the Earth's center
    simd_double3 side = simd_cross(axis, simd_make_double3(0, 1, 0));
    if (simd_length(side) < 1e-3) side = simd_cross(axis, simd_make_double3(1, 0, 0));
    const double rpx = simd_length(LodePx(proj, simd_normalize(side), sz) - pc); // the Earth's radius on screen
    simd_double2 dir = LodePx(proj, m * 0.6, sz) - pc;
    dir = simd_length(dir) > 2.0 ? simd_normalize(dir) : simd_normalize(simd_make_double2(-0.6, -0.8));
    const double mr = fmin(44.0 * scale, fmax(18.0 * scale, 0.19 * rpx));
    const double margin = 18.0 * scale;
    const double ax = sz.width * 0.5 - margin - mr;
    const double ay = (dir.y > 0 ? sz.height * 0.5 - 210.0 * scale : sz.height * 0.5 - 70.0 * scale) - mr;
    const double fit = (ax > 0 && ay > 0) ? 1.0 / sqrt(pow(dir.x / ax, 2) + pow(dir.y / ay, 2)) : 0;
    const double t = fmin(rpx * 1.32 + mr, fit);
    if (t > 0) {
      const simd_double2 pm = pc + dir * t;
      const simd_double3 c = LodeRay(inv, pm, sz);
      const double cosr = simd_dot(c, LodeRay(inv, pm + simd_make_double2(mr, 0), sz));
      const simd_double3 sunD = simd_make_double3(u.sun.x, u.sun.y, u.sun.z);
      const simd_double3 s2 = LodeTurn(sunD, m, c), n2 = LodeTurn(simd_make_double3(0, 1, 0), m, c);
      u.moonC = simd_make_float4((float)c.x, (float)c.y, (float)c.z, (float)cosr);
      u.moonSun = simd_make_float4((float)s2.x, (float)s2.y, (float)s2.z, simd_dot(m, axis) < 0 ? 1.0f : 0.0f);
      u.moonNorth = simd_make_float4((float)n2.x, (float)n2.y, (float)n2.z, (float)moonShow);
      // the photo's detail for the Moon's size: half its width (one hemisphere) across the Moon's width in pixels
      u.moonMore.x = (float)fmax(0.0, log2((double)_moon.width * 0.5 / (2.0 * mr)));
    }
  }
  [encoder pushDebugGroup:@"Lode atmosphere"];
  [encoder setRenderPipelineState:_pipeline];
  [encoder setDepthStencilState:_depthStencil];
  [encoder setFragmentBytes:&u length:sizeof(u) atIndex:0];
  [encoder setFragmentTexture:(_lights ?: _noLights) atIndex:0];
  [encoder setFragmentTexture:(_moon ?: _noLights) atIndex:1];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [encoder popDebugGroup];
#endif
}

@end
