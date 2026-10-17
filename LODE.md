# Lode's MapLibre Native

Lode Maps' copy of [MapLibre Native](https://github.com/maplibre/maplibre-native) (BSD 2-Clause, see LICENSE.md),
started from `ios-v7.0.0-pre-globe.0` (branch `lode`). Branch `lode-next` is MapLibre's newer globe work (PR 4533, Oct 16: dragging keeps the spot under your finger, smooth flings, no tile flicker) with our build on top.

- Build: `gh workflow run lode-ios.yml --repo helass457/lode-maplibre -f label=lode-N` (about an hour on GitHub's Macs).
  It publishes `MapLibre.dynamic.xcframework.zip` and its checksum as release `lode-N`.
- Use: point `helass457/lode-maplibre-dist`'s Package.swift at that release (url + checksum) and tag it, then set
  `nativeVersion` (and the package URL) in lode-app's app.json.
- Upstream fixes: `git fetch https://github.com/maplibre/maplibre-native main` and merge or cherry-pick.
