**Full thread handover for Claude** (ready to copy-paste)

### Context
This is a recent X/Twitter thread (posted 22 Oct 2026) by **@ostrisai** (Ostris), an independent AI/ML researcher and creator of the open-source **AI Toolkit**.  

He proposes a new training target for **flow-matching diffusion models** called **“x0 velocity”**.  

Key claims:
- Keeps the exact same forward process and sampling trajectory as standard flow matching.
- Network output is converted back to velocity with a single subtraction (compatible with existing samplers/schedulers/guidance).
- Target formula:  
  `target = noisy - (noise - clean) = clean + (1 - t) * (clean - noise)`
- At pure noise (t=1) it is plain x0 prediction.  
- Produces an **image-shaped** output at every timestep.  
- The residual (input → target) has **constant magnitude** across all timesteps (unlike plain x0, whose residual shrinks to zero near the clean end, or pure velocity, whose residual grows).  
- Loss is exactly the unweighted flow-matching velocity loss (no timestep weighting needed).  
- Claims more stable training and better tolerance for high-frequency content / deeper latent compression / more latent channels.

He shows early qualitative results converting the MOSS image model into a **64× latent compression** model (using patch size 8 instead of the original patch 2), trained for only 1 day on a single GPU.

### Full Thread

**Main post** (@ostrisai, 13:18 UTC)  
I think there is a better training target for flow-matching diffusion models. I call it **x0 velocity**.

It keeps the forward process and the sampling trajectory of flow matching exactly. The network's output is converted back to a velocity with one subtraction, so any flow-matching sampler, scheduler, or guidance method runs unchanged. What changes is the target the network regresses onto:

```
target = noisy - (noise - clean)  =  clean + (1 - t) * (clean - noise)
```

At the pure-noise end this is plain x0 prediction. Below that it is x0 extrapolated one unit along the flow line. The output is image-shaped at every timestep, never a noise field, and the correction the network has to make, the difference between its input and its target, has the same magnitude at every timestep. Plain x0 prediction's correction shrinks to zero at the clean end, which starves the fine-detail steps. Velocity prediction's correction grows toward the clean end, where the network has to reproduce and invert its own input. x0 velocity sits between them: a constant-size correction with an image-like output, and a loss that is exactly the flow-matching velocity loss, so it needs no timestep weighting.

In my testing it trains more stably than either, and it is more tolerant of high-frequency content, which lets models perform better with more latent channels and deeper compression. 🧵

*(Attached: diagram comparing Flow matching / x0 prediction / x0 velocity targets + MSE values across sigma levels. The x0 velocity residual magnitude stays constant.)*

**Follow-up post** (@ostrisai, immediately after)  
I am converting MOSS image into a 64x latent compression model. I am applying a patch 8 to the latent instead of the patch 2 it has. It is still training, these are both 1 day on a single GPU, but here are current samples. First is flow-matching, second is x0 velocity.

*(4 images attached)*  
- Image 1 & 3: standard flow-matching results (noticeably blurrier / more artifacts)  
- Image 2 & 4: x0 velocity results (significantly sharper and more coherent)

### Notable replies

**@levzzz5154**:  
Just Image Transformers did this  
https://arxiv.org/abs/2511.13720  
*(Paper: “Back to Basics: Let Denoising Generative Models Denoise” by Tianhong Li & Kaiming He – advocates pure x0 / clean-image prediction with large-patch Image Transformers)*

**@ostrisai**:  
Oh nice! I will give it a read, thank you.

**@SwayStar123**:  
@grok How is this different from JiT/Just Image Transformer?

**@grok** (reply):  
JiT uses pure x0 prediction: the net outputs a clean-image estimate, then converts to velocity via division by (1-t). That pairs with a reweighted velocity loss.

x0 velocity instead regresses onto the extrapolated target `clean + (1-t)(clean-noise)`. The output stays image-shaped, converts to velocity by a single subtraction (no division), and the loss is exactly the unweighted flow-matching velocity loss with constant residual magnitude at every t. It is a hybrid parameterization, not the pure x0 target JiT uses.

**Later reply by @ostrisai** (regarding converting existing models):  
The model has to be trained to use it. X0 models can pick it up pretty quickly, such as Hidream o1. Flow-matching models need their output negated. But they can both be warm started and only need a little bit of fine-tuning to convert them.

---

### Additional notes for Claude
- Author is practical / engineering-oriented (runs AI Toolkit). He explicitly says he doesn’t usually write papers.
- The method is presented as a simple drop-in change to the training target while remaining fully compatible with existing flow-matching infrastructure.
- Early qualitative results look promising for high-compression regimes, but no quantitative metrics (FID, etc.) or large-scale ablations are provided yet.

You now have the complete relevant thread + technical summary.