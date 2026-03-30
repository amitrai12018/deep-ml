import numpy as np
def translate_object(points, tx, ty):
	points=np.array(points)
	path=np.array([tx,ty])
	points+=path
	return points
	return translated_points
