#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import pyoclnbody.context

matplotlib.use('QT5Agg')

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ctx = pyoclnbody.context.ParticleSystem()

for i in range(250):
    ctx.set_data()
    pl = ctx.get_data()
    if i % 25 == 0:
        ax.scatter([p['x'] for p in pl], [p['y'] for p in pl], [p['z']
                                                                for p in pl])
        plt.draw()
        plt.show()
        plt.pause(0.1)

plt.waitforbuttonpress()