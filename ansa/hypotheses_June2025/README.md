# 7 simulations w/ and w/o softening (14 total)
NOTE: Look at what counts are semi-major and semi-minor axis, is it a problem that x-axis is always semi-major (it can be shorter!)
##### properties of all sims
- no cell migration from left side (new cells are only introduced through division)
- division is random - cell picked to divide is random, not near the epithelium
- all sims have elastic links from boundary points to the nearest cell
## sims (do each twice, once with and once without softening!!):
1. random motion, random division
	- all have random motion term eta
	- angle of division - uniform distribution $(0, 2\pi)$
2. random motion, directed division (PD axis)
	- div angle  $0\pm \frac{\pi}{6}$ standard deviation
3. random motion, directed division (AP axis)
	- div angle  $\frac{\pi}{2}\pm \frac{\pi}{6}$ standard deviation
4. random motion except X% moving towards distal random division 
	- see chick paper
5. my favorite with side stiffening
	- instead of softening (dividing kb by something), stiffening multiplied kb by something
6. softening on the sides
7. intercalation
8. Local Division (near boundary only)
## file name structure
agentslimbreg[sim_number][s] for soft
agentslimbreg[sim_number] for not soft

[Link to powerpoint](https://northeastern-my.sharepoint.com/:p:/r/personal/brewsmith_a_northeastern_edu/Documents/sims_comparison.pptx?d=w04c35423ba5740f1863ee04b2f4e500e&csf=1&web=1&e=HLBYl4)