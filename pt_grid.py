import numpy as np

class PT_Grid:
    def __init__(self, 
                 log_P_min=-5, log_P_max=2, log_P_step=1,
                 T_tuples=[(300, 3000, 100)], 
                 P_grid=None, T_grid=None, # optionally pass in P and T grid values
                 file='PTpaths_high_test.ls'):
        """
        Initialize the PT_Grid with parameters for the pressure and temperature grid.
        """
        self.log_P_min = log_P_min
        self.log_P_max = log_P_max
        self.log_P_step = log_P_step
        self.T_tuples = T_tuples
        self.file = file
        self.P_grid = P_grid
        self.T_grid = T_grid

    def make_grid(self):
        """
        Generate the grid of P and T values and save them to a file.
        """
        # Generate pressure grid
        self.P_grid = 10.0 ** np.arange(self.log_P_min, self.log_P_max + self.log_P_step, self.log_P_step)
        
        # Generate temperature grid
        T_grid_list = []
        for T_i in self.T_tuples:
            T_min, T_max, T_step = T_i
            T_grid_list.append(np.arange(T_min, T_max + T_step, T_step))
        self.T_grid = np.concatenate(T_grid_list)
        
        print(' ** PT_Grid **')
        # Print grid size
        print(f' Size of grid: {self.P_grid.size * self.T_grid.size}')
        
        # Save grid to file
        self.save_to_file()
        
        print(f' Pressure: {self.P_grid}\n Temperature: {self.T_grid}')
        print(f' Saved to {self.file}')
        
        return self.P_grid, self.T_grid

    def save_to_file(self):
        """
        Save the P and T grid values to the specified file.
        """
        with open(self.file, 'w') as f:
            for p in self.P_grid:
                for t in self.T_grid:
                    # Separate with 4 spaces
                    f.write(f'{p:.1e}    {t:6.1f}\n')
    
    @classmethod
    def load_from_file(cls, file):
        """
        Load the P and T grid values from the specified file.
        """
        data = np.genfromtxt(file).T
        P_grid = data[0]
        T_grid = data[1]
        return P_grid, T_grid
        
    
if __name__ == '__main__':

    # Usage
    # Create custom grid with P and T values
    # define several temperature ranges with different spacings
    import pathlib
    file = pathlib.Path('input_data/PT_grids/PTpaths_high_dario_test.ls')
    file.parent.mkdir(parents=True, exist_ok=True)
    
    pt_grid = PT_Grid(
        log_P_min=-5., log_P_max=2., log_P_step=1.,
        T_tuples=[(1200, 1600, 400), # (T_min, T_max, T_step)
                 (1800, 5000, 200)],# (T_min, T_max, T_step)
        file=file
    )
    pt_grid.make_grid()


    # Load the grid from file
    P_grid, T_grid = PT_Grid.load_from_file(file)
    print(f' Loaded P_grid: {P_grid}, T_grid: {T_grid}')
    
    # unique values
    print(f' Unique P_grid: {np.unique(P_grid)}\nUnique T_grid: {np.unique(T_grid)}')
