# Curved Lane Line Detection

This project contains baseline for curved lane lines detection. 
But there are some cases, when this approach works bad:
<ul>
<li>When road turns relatively hard (sharp turn or the road is winding).</li>
<li>When the road gets sunshined or other light sources.</li>
</ul>

The first case appears because of <i>perspective_warp()</i> returns kind of cropped image. 
When the road is winding, lane line often go out of warped image.<br>

Second case appears during <i>Sobel</i> operation. 
Sobel operator detects gradients of sunshine borders and it is noise in that case.

# Links
<ol>
<li>Curved Lane Detection. URL: https://www.hackster.io/kemfic/curved-lane-detection-34f771.</li>
</ol>