class MovingObject:
    def __init__(self, obj_id, dist_x, dist_y):
        self.obj_id = obj_id
        # distance from camera x and y
        self.dist_x = dist_x
        self.dist_y = dist_y

        # counting the number of frames in which the object appears
        self.frame_counter = 0

        # final speed of the object (in m/s)
        self.speed = 0

        # it maintain the points of the trajectory of the object during the frames
        self.trajectory = []

        # speed on x and y axis (in m/s) for debug.
        self.speed_x = 0
        self.speed_y = 0

        # to compute the average speed
        self.speed_counter = 0


class MODictionary:
    def __init__(self):
        self.dict = {}

    def add(self, obj):
        self.dict[str(int(obj.obj_id))] = obj

    def get(self, obj_id):
        return self.dict[str(int(obj_id))]

    def id_in(self, obj_id):
        return str(int(obj_id)) in self.dict.keys()
