import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_targetLines(origin,n_target_lines, n_steps, line_lenght=0.2):
    """ Generate n_steps targets points lying on a 2D line for each n_target_lines 
        Args:
            origin: initial (x_0,y_0)-coords for all lines
            n_target_lines: n. of target lines 
            n_steps: n. of points in each target line
            line_lenght: lenght of each target line
        Returns:
            torch.tensor containing list of points of each target line, shape: [n_target_lines, n_steps]
    """

    # Generate n. target lines 
    step_size = line_lenght / n_steps
    radiuses = np.linspace(step_size, line_lenght+step_size, n_steps)

    # Create n. equally spaced angle to create lines
    ang_range = np.arange(0,n_target_lines) * (2 * np.pi) / n_target_lines

    # Use unit circle to create lines by multiplying by growing radiuses to draw lines
    cos_targ_xs = np.cos(ang_range)
    sin_targ_ys = np.sin(ang_range)
    x_targ = []
    y_targ = []
    # Use for loop to multiple each radius by corrspoding sine and cosine (should also be doable by proper np broadcasting)
    for r in radiuses: 
        x_targ.append((cos_targ_xs * r)+origin[0])
        y_targ.append((sin_targ_ys * r)+origin[1])

    x_targ = torch.tensor(np.array(x_targ),dtype=torch.float32).T # shape:[batch, n_steps]
    y_targ = torch.tensor(np.array(y_targ),dtype=torch.float32).T # shape:[batch, n_steps]

    ## ======== Verification Plot ===========
    # Check the targets are on 6 different lines
    #plt.plot(x_targ,y_targ)
    #plt.show()
    ## =============================

    return x_targ, y_targ

