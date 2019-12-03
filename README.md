# Monte-Carlo-CUDA-Pi-Simulation
📉🥧Monte Carlo Pi Simulation using CUDA Primitives
  
View at:  
<a href="https://godbolt.org/z/cDfQwi"><img height="50" width="165" src="http://fragata.arcos.inf.uc3m.es/dist/assets/site-logo.3f5bcf90b56ade7be40ffa8cca8b2056.svg"></a>
### Compiling
* When compiling you have to link against the curand library
```
nvcc monte_carlo_pi_simulation.cu -lcurand
```
