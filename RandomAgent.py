import retro
import gym
#creates an environment with airstriker-genesis game
env = retro.make(game = "Airstriker-Genesis")
#Creates an observation
obs = env.reset()
#Prints the shape of our observation
print(obs.shape)

done = False

while not done:
    #we grab observer and done and info
    #Going to take a random action from the sample space. Then will pass back some information
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
env.close()
