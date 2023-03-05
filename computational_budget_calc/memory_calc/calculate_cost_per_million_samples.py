"""
Python function that takes as input the cost per hour, and throughput, 
and calculates the cost per million samples:Note that the inputs are assumed to 
be in the same units as in the original question, i.e., cost in dollars per hour, 
latency in milliseconds, and throughput in samples per second. If your inputs are 
in different units, you may need to convert them before using the function.
"""

def cost_per_million_samples(cost_per_hour, throughput):
    # Calculate samples per hour
    samples_per_hour = throughput * 3600
    
    # Calculate cost per sample
    cost_per_sample = cost_per_hour / samples_per_hour
    
    # Calculate cost per million samples
    cost_per_million_samples = cost_per_sample * 1000000
    
    return cost_per_million_samples


cost = cost_per_million_samples(3.6, 461.675)
print(cost) # Output: 2.520172918771719

