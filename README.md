# Idea Dump


## Cheap Neuron-Resolution Brain Scanning (physics, optics)
TLDR: an idea to exploit properties of diffraction patterns to scan a small object at extremely high resolution using a cheap camera sensor. 

1. Current high-resolution brain scanning approaches are slow and expensive. Ideally there would be a scanning method that could scan a brain in about 20 mins to 1 micron resolution for essentially zero cost.

2. If you shine a laser through a target with spatial structure on the order of 1-100 microns, a diffraction pattern is projected outwards. 

3. This pattern has two properties: it contains some of the information of the spatial structure of the target, and it's always in focus. 

4. Here is the bit that makes this idea potentially interesting: since the pattern is always in focus, you can take the lens off a normal camera CMOS sensor and scan the exposed sensor over the projected pattern with a 2D linear rail, then stitch the video into a giant rectangular image. Now you can use a ~10 megapixel sensor to record an enormous amount of data (100 gigapixels would be a very conservative estimate), which you can then process back into the original spatial structure to get an image of the target with extremely high spatial resolution and enormous field of view (some people say you need both the amplitude and phase of the diffraction pattern to reconstruct the spatial image -- you can actually get a near-perfect image from just the intensity using [iterative methods](https://arxiv.org/pdf/1203.4756.pdf)! Add a neural net image prior and I think you'd be hard pressed to see any distortion).

5. This only records a single perspective through the sample, but you can rotate the sample about its central axis to record multiple perspectives, and use standard tomographic reconstruction algorithms to get out a 3D structure. You'll need a high-quality beam, so will probably have to use e.g. a spatially-filtered HeNe laser.

Maybe you could use this to scan a Drosphilia brain. Best case you only get the structure, not the synaptic weights or neurotransmitter concentrations... but ya gotta start somewhere!

See: [comments from Adam](https://twitter.com/AdamMarblestone/status/1060259368246145024).

## Nuclear Fusion: Plasma Simulation with Learned Dynamic Approximation (physics, ai)
Fusion research is in this weird place where there are lots of possible approaches, but nobody knows for sure if one is going to work ahead of time because nobody has an accurate simulator for the plasma regime they are in. This means that researchers have to *spend a decade and $10-500M building a machine before they know if it will work*.

Modelling plasma is really really hard. Plasma has structure on many different spatial and temporal scales, and often you decide upon a level of abstraction to simulate at but get completely screwed over when you build the experiment in real life because your simulator abstraction inadvertently missed out some weird plasma phenomenon that makes your machine not work (see: NIF, JET, TFTR, probably ITER etc etc).

The lowest level of abstraction is [modelling each particle](http://www.ss.ncu.edu.tw/~lyu/lecture_files_en/lyu_SPP_Book_A4format_pdf_html/pdf_1_Ch/lyu_SPP_Chapter_2.pdf) (the Klimontovich Description) which is computationally impossible for a large plasmas even using the world's biggest supercomputer, but would probably work pretty well if you actually had the resources.

There's been lots of recent work on compressing cfd simulators ([see: cool spacex video on wavelet compression](https://www.youtube.com/watch?v=txk-VO1hzBY)), which might be a useful direction (I don't think people are even doing this right now), but I think the really paradigm-shifting approach would be to *learn what approximations you can make (with deep learning) to keep the simulation both efficient and high-fidelity*. i.e. start from the Klimontovich Description, and have a collection of neural nets that dynamically adjust the grain-size of the approximation (cell size, superparticle size, timestep, energy spectrum precision, interaction length etc etc) for different regions of space and time, trained to make the decompressed simulator evolve identically to the full simulation. Your training data (full Klimontovich for small plasmas) would be expensive to obtain, but I think the outcome would be worth it, and you might not actually need that much of it.

Note: I don't mean 'train a 3D recurrent CNN to model a numerical simulator' -- I mean 'train a 3D recurrent CNN to dynamically control cell size/particle size/ timestep of a numerical simulation so that it outputs the right thing with minimum resources'.


## Fast Human Deep Space Travel (physics)
To get anywhere fast, you'd have to accelerate for half the journey, and deccelerate for the second half. The problem is that humans degrade unless stored at 1g, which means you could never accelerate at more than 1g. Sounds slow!

A way around this would be to surround the living quarters with a huge superconducting electromagnet, and [diamagnetically levitate the water in the human body](https://www.youtube.com/watch?v=A1vyB-O5i6E) at one less g than your acceleration. Now you can reach 20g or so before you start getting damaged by the [emf across your blood](http://www.dartmouth.edu/~sshepherd/research/Shielding/docs/Schenck_00.pdf). Obviously this only matters if we're still implemented on meat at that point.


## Neural Net Inference at the Speed of Light (physics)
It turns out that you can [learn how to etch a silicon optical waveguide](https://arxiv.org/pdf/1504.00095.pdf) to do various cool things. It would be cool to etch a silicon chip to perform a neural net inference (e.g. a [basic phoneme task](https://arxiv.org/abs/1610.02365)) at the speed of light: shine in the inputs at one end, get out the result at the other. The Maxwell FDFD simulator is linked in the paper, so you can generate all the training data you need. Problem is that you could only implement linear functions and I can't think of an obvious way to implement a zero-power ReLU. Maybe you could etch a diffractive neural net into a pumped laser crystal...


## Finding Wolfram's Universal Rule (physics)
Steven Wolfram [thinks](http://blog.stephenwolfram.com/2017/05/a-new-kind-of-science-a-15-year-view/) that the universe is implemented on a cellular automaton with a simple rule. He said somewhere that he thinks the only way to figure out the rule is to literally try them all out, and see which one generates the laws of physics. Cellular automata are equivalent to recurrent convnets (they both impose exactly the same prior: that the system's underlying laws are spatially quantized, local, and translationally invariant), which you can train with backprop (various hacks let you do this even with discrete values). 

You could use this approach to learn Wolfram's universal rule from data: make a convnet that simulates the entire world in 4D. If you can get the loss to zero then the filters will automatically represent the rule.



## Hierarchical Imitation Learning (AI)
TLDR: how to automatically decompose a long multitask expert demonstration into subtasks, in an unsupervised way. As of 2018-11-10 this is an unsolved problem. I'm 99% sure this would work for simple tasks, like grasping stuff on a table, and about 80% sure it would work (with some modification) for complex stuff like DOTA 2/starcraft/Montezuma's Revenge.

With sufficient time and compute, it’s possible to solve incredibly complex problems from scratch with RL. However we do not always have these luxuries... Good reward shaping can make the learning process substantially more efficient, but it becomes harder and harder as the complexity of the problem grows (and can be quite an art, especially out of simulation).

Bootstrapping the learning process with imitation learning can substantially lower the amount of rollouts needed for mastery, but traditional IL methods are generally limited to single, well-defined tasks. An ideal system would absorb thousands of hours of unstructured multitask demonstrations of dextrous tasks (e.g. demonstrators manipulating common household objects with their hands), automatically segment them into their constituent subtasks, and embed everything into an enormous learned latent space of sub-policies/options which you could then hook up to a higher-level control policy. Some recent work from Berkeley has done something similar, but it still requires a user-specified dictionary of subtasks (which isn't a particularly scalable approach).

No current work has demonstrated how to automatically figure out how to segment subtasks, without a user-specified subtask dictionary. Here's how you could figure out how to segment, with no prior information (you would obviously call it Automatic Subtask Segmentation for acronymic reasons):

Gather giant dataset of multitask demonstrations (e.g. ten hours of a demonstrator picking up random objects on a table). Train a single small RNN to imitate the expert's actions over the entire time series. **Look at the timesteps where the imitation loss peaks: these will be the natural points to segment the subtasks.** Cut the demonstrations along these boundaries, then embed them into a giant hierarchical policy.

Also feels vaguely cognitively plausible: everything functions on autopilot until the brain's internal predictive model disagrees with reality, then high-level attention gets summoned to decide what high-level action to take next.

Could perhaps also apply to an already-learned monolithic RNN policy (e.g. openai five), to cast into a hierarchical model for increased transferrability/learning speed).



## Compact High-Resolution Tactile Sensing (robotics, physics)

Tactile sensing is very important for robotics, but right now there isn't a high-resolution tactile sensor that can fit inside a fingertip. The [GelSight sensor](https://ruili.io/publications/rui_iccp13_slides.pdf) is closest. It's a high-resolution tactile sensor used by Google for robotics research, but has a width of about 5cm.

Most of the size of the current sensor is just focal length for the lens. You could replace the lens with a pinhole camera to scale everything down to fit inside a fingertip.

Cameras still feel hacky, though... Ideally you'd get some material with a resistance log-response to pressure (human skin can detect many orders of magnitude of pressure), and attach a grid of terminals to measure resistance, then get a tactile map out by interpolating the resistance values. Could sample at high spatial resolution at fingertips, and low at elbows etc. Material could be [Quantum Tunnelling Composite](https://www.youtube.com/watch?v=J-vrjdvi94w&t=0m35s) (which despite the cool name is just a zinc polymer ink with a log-response to pressure) fabric embedded in silicone. edit: someone [just did it](https://resou.osaka-u.ac.jp/en/research/2018/20181114_2), but not with QTC fabric.





## Long-Distance Penetrative Sensing (physics)
Extend [this work](http://rfpose.csail.mit.edu/) using ultra-wideband radio pulses, an array of a few thousand 3D printed UWB antennas and a phased-array scanning beam. You should be able to penetratively scan an entire building to a resolution of one cm or so. Sweet dreams, citizen.

