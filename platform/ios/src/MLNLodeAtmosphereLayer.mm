#import "MLNLodeAtmosphereLayer.h"
#import "MLNMapView.h"
#import "MLNBackendResource.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

// One full-screen triangle. Each pixel's view ray is taken back through the globe's projection into unit-sphere space:
// a ray that passes within 1 of the center hits the Earth (left alone, but for a faint rim of light at its edge);
// anything else is space, lit by a glow that fades with the distance from the edge, with sparse still stars.
static const char *kLodeAtmosphereShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LodeAtmoUniforms {
  float4x4 inv;     // clip space -> unit sphere space
  float2 size;      // drawable size in pixels
  float alpha;      // how much of it shows (the globe's projection transition)
  float scale;      // pixels per point
  float4 sun;       // xyz: toward the sun, in unit-sphere space (x 90°E, y north, z null island)
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

fragment float4 lodeAtmoFragment(LodeAtmoOut in [[stage_in]], constant LodeAtmoUniforms &u [[buffer(0)]]) {
  float2 ndc = float2(in.pos.x / u.size.x * 2.0 - 1.0, 1.0 - in.pos.y / u.size.y * 2.0);
  float4 a = u.inv * float4(ndc, -0.5, 1.0);
  float4 b = u.inv * float4(ndc, 0.5, 1.0);
  float3 o = a.xyz / a.w;
  float3 d = normalize(b.xyz / b.w - o);
  float dist = length(o - dot(o, d) * d); // how close this pixel's ray passes to the Earth's center (radius 1)
  if (dist < 1.0) {
    // on the Earth: the side in darkness right now is shaded, with a soft edge between day and night (where the ray
    // first meets the sphere, its height above the sun's horizon there); a faint rim of light at the edge
    float3 p = o + (-dot(o, d) - sqrt(max(0.0, 1.0 - dist * dist))) * d;
    float sunUp = dot(normalize(p), u.sun.xyz);
    float night = smoothstep(0.08, -0.12, sunUp) * 0.55;
    float rim = smoothstep(0.94, 1.0, dist) * 0.28;
    float a = max(night, rim);
    if (a <= 0.003) discard_fragment();
    float3 col = rim > night ? float3(0.55, 0.74, 1.0) : float3(0.0, 0.01, 0.035);
    return float4(col, a * u.alpha);
  }
  float g = dist - 1.0;
  float glow = clamp(exp(-g * 6.5) * 0.75 + exp(-g * 28.0) * 0.45, 0.0, 1.0);
  float3 col = mix(float3(0.010, 0.013, 0.026), float3(0.42, 0.64, 1.0), glow);
  // stars: one at most per 4-point cell, at a random spot in it, a small round point (soft edge, varied size and
  // brightness, a few warm or cool), fading out in the glow
  float cellPx = 4.0 * u.scale;
  float2 cell = floor(in.pos.xy / cellPx);
  float h = lodeHash(cell);
  if (h > 0.972) {
    float2 spot = (cell + float2(lodeHash(cell + 17.0), lodeHash(cell + 41.0)) * 0.8 + 0.1) * cellPx;
    float k = (h - 0.972) / 0.028;
    float r = (0.35 + 0.55 * k * k) * u.scale;
    float lit = smoothstep(r, r * 0.25, length(in.pos.xy - spot)) * (0.35 + 0.65 * k);
    float tint = lodeHash(cell + 7.0);
    float3 star = tint > 0.9 ? float3(1.0, 0.86, 0.7) : tint < 0.1 ? float3(0.75, 0.85, 1.0) : float3(1.0);
    col += star * lit * (1.0 - clamp(glow * 2.2, 0.0, 1.0));
  }
  return float4(col, u.alpha);
}
)METAL";

typedef struct {
  simd_float4x4 inv;
  simd_float2 size;
  float alpha;
  float scale;
  simd_float4 sun;
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

@implementation MLNLodeAtmosphereLayer {
  id<MTLRenderPipelineState> _pipeline;
  id<MTLDepthStencilState> _depthStencil;
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
#endif
}

- (void)willMoveFromMapView:(MLNMapView *)mapView {
  _pipeline = nil;
  _depthStencil = nil;
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
  [encoder pushDebugGroup:@"Lode atmosphere"];
  [encoder setRenderPipelineState:_pipeline];
  [encoder setDepthStencilState:_depthStencil];
  [encoder setFragmentBytes:&u length:sizeof(u) atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [encoder popDebugGroup];
#endif
}

@end
