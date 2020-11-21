import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# List to keep track of frames
total_frames = []

class Frame:
    
    def __init__(self, name, color=[0,0,0], parent_frame=None, is_point=False):
        """
        If parent frame is not specified then the non-inertial 
        global frame is taken as parent.
        Parent frame is the one, with respect to which, you are
        defining the current frame
        
        """
        
        self.name = name
        self.color = [color, [1,0,0], [0,1,0], [0,0,1]]
        self.transform = np.eye(4)
        
        # Keeping a reference to parent frame
        self.parent = parent_frame
        
        if not self.parent is None:
            if self.parent.is_point:
                # Cannot make a point as a parent frame of any other frame
                raise Exception("Point " + self.parent.name + " is a parent frame for "
                       + self.name + " which is not allowed.\nPoint as a parent frame is not allowed")
        
        self.is_point = is_point
            
        # Anchors have 1 as forth value because of homogenous coordinates
        self._anchors = np.array([
            [0, 0, 0, 1], # Frame Origin
            [1, 0, 0, 1], # x-axis
            [0, 1, 0, 1], # y-axis
            [0, 0, 1, 1]  # z-axis
        ])
        
        global total_frames
        total_frames.append(self)
        
    def rotate_x(self, angle):
        """ 
        Provide angle in degrees (for user simplicity) 
        
        """
        angle = np.deg2rad(angle)
        self.transform[:3,:3] = self.transform[:3,:3] @ np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        
    def rotate_y(self, angle):
        """ 
        Provide angle in degrees (for user simplicity) 
        
        """
        angle = np.deg2rad(angle)
        self.transform[:3,:3] = self.transform[:3,:3] @ np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
    def rotate_z(self, angle):
        """ 
        Provide angle in degrees (for user simplicity) 
        
        """
        angle = np.deg2rad(angle)
        self.transform[:3,:3] = self.transform[:3,:3] @ np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
    def translate(self, x, y, z):
        """
        Provide translation in x, y and z
        
        """
        
        self.transform[:3,-1] = np.array([x, y, z])
        
    def get_transformed_anchors(self):
        if self.parent is None:
            return self._anchors.dot(self.transform.T)
        else:
            return self._anchors.dot((self.parent.get_propagated_transform() @ self.transform).T)
        
    def get_propagated_transform(self):
        if self.parent is None:
            return self.transform
        else:
            return self.parent.get_propagated_transform() @ self.transform

    def as_seen_from(self, frame):
        if frame.is_point:
            raise Exception("Entities can only be viewed as seen from a frame\nYou have passed a point as a frame")

        transform = np.linalg.inv(frame.get_propagated_transform()) @ self.get_propagated_transform()
        transformed_entity = self._anchors.dot(transform.T)
        
        transformed_entity = transformed_entity[0,:3]

        print("Transformation:")
        print(transform)
        print("Transformed Entity origin:")
        print(transformed_entity)


# Global Functions
def clear_frames():
    global total_frames
    total_frames = []

def show_frames():
    fig = plt.figure()
    fig.clear()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    global total_frames
    for frame in total_frames:
        frame_anchors = frame.get_transformed_anchors()
        # Drawing anchors that represent this frame

        if frame.is_point:
            # Scatter point
            ax.scatter3D(frame_anchors[0,0],
                        frame_anchors[0,1],
                        frame_anchors[0,2], c=frame.color[0])
        else:
            # Scatter anchors
            ax.scatter3D(frame_anchors[:,0],
                        frame_anchors[:,1],
                        frame_anchors[:,2], c=frame.color)

            # Drawing lines between anchors
            for i in range(1, 4):
                ax.plot([frame_anchors[0,0], frame_anchors[i,0]],
                        [frame_anchors[0,1], frame_anchors[i,1]],
                        [frame_anchors[0,2], frame_anchors[i,2]], c=frame.color[i])

        # Drawing line from parent frame to this frame to inidicate
        # Parent-Child relationship
        if not frame.parent is None:
            parent_frame_anchors = frame.parent.get_transformed_anchors()
            ax.plot([parent_frame_anchors[0,0], frame_anchors[0,0]],
                    [parent_frame_anchors[0,1], frame_anchors[0,1]],
                    [parent_frame_anchors[0,2], frame_anchors[0,2]], 
                    c=frame.parent.color[0], linestyle='dashed')

        entity_type = "Point: " if frame.is_point else "Frame: "

        # Annotating frame
        ax.text(frame_anchors[0,0], 
                frame_anchors[0,1],
                frame_anchors[0,2], entity_type + frame.name, color=frame.color[0])

    # Display all the plotted frames at the end
    set_aspect_equal_3d(ax)
    plt.show()

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])