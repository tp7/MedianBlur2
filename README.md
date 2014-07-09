## MedianBlur2 ##

Implementation of [constant time median filter](http://nomis80.org/ctmf.html) for AviSynth. 

### Usage
This plugin provides two functions: `MedianBlur(int radiusy, int radiusu, int radiusv)` is a spatial-only version and `MedianBlurTemporal(int radiusy, int radiusu, int radiusv, int temporalRadius)` is a spatio-temporal version.

Maximum radius on every plane is limited to 127.

### Performance
Unlike the old MedianBlur, this implementation has constant runtime complexity, meaning that theoretical performance is same for any radius. A few additional optimizations are included:

1. Radii 1 and 2 are special-cased (a lot faster but less generic algorithm) when SSE2 is available.
2. For 2 < radius < 8, generic approach with 8-bit bin size is used.
3. For large radii, 16-bit bins are used, as described in the paper. 

In other words, you can expect huge performance drop when going from 1 to 2, not so huge but still large from 2 to 3 and a noticeable slowdown from 7 to 8. Between them the fps should be constant and it actually might get a bit faster with larger radii. 

Performance with radius > 2 also depends on the actual frame content. Processing colorbars() is a lot faster than addgrainc(10000).

### License ###
This project is licensed under the [MIT license][mit_license]. Binaries are [GPL v2][gpl_v2] because if I understand licensing stuff right (please tell me if I don't) they must be.

[mit_license]: http://opensource.org/licenses/MIT
[gpl_v2]: http://www.gnu.org/licenses/gpl-2.0.html
