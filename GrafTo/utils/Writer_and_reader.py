import numpy as np
import MDAnalysis as mda
import pandas as pd
import json
from pathlib import Path

class WriterAndReader:
    
    def __init__(self, parent_system) -> None:
        self.parent = parent_system
        self.root_dir = parent_system.root_dir
    
    @property
    def universe(self):
        """Always gets current universe from parent"""
        return self.parent.universe
        
    def out_gro(self, name="initial_config.gro", u=None, outSizes=False, convert=1):
        if u:
            universe = u
        else:
            try:
                universe = self.universe
            except AttributeError:
                raise Exception("No universe found")
            
        filename = f"{self.root_dir}/{name}"
        universe.atoms.positions = np.array(universe.atoms.positions)*convert
        universe.dimensions[:2] = universe.dimensions[:2]*convert
        universe.dimensions[2] = universe.dimensions[2]+200*convert
        gro_writer = mda.coordinates.GRO.GROWriter(filename, n_atoms=len(universe.atoms))
        gro_writer.write(universe)
        gro_writer.close()

        if outSizes:
            out = open(f"{self.root_dir}/molSizes.dat", "w")
            for m in self.molSizes:
                out.write(f"{m}\n")
            out.close()

    def read_inputs_assembler(self, path_to_input_file):
        """
        Reads the inputs from a file or dictionary.

        Parameters:
        - path_to_input_file: The path to the input file or a dictionary.

        Raises:
        - Exception: If the input file is not a path to a .json file or a dictionary.
        """
        if isinstance(path_to_input_file, str):
            import json
            inputs = json.load(open(path_to_input_file,"r"))
        elif isinstance(path_to_input_file, dict):
            inputs = path_to_input_file
        else:
            raise Exception("Input file must be a path to a .json file or a dictionary")

        self.parent.root_dir = inputs["root directory"]
        self.parent.blocks = inputs["blocks"]
        self.parent.positions = np.array(inputs["positions"])
        self.parent.transforms = [None]*len(self.blocks) if not inputs["transforms"] else inputs["transforms"]
        self.parent.system_name = inputs["system name"]
        self.parent.out_name = inputs["out name"]
        self.parent.box = inputs["box dimensions"]

        print(f"\nname: {self.parent.system_name}\nroot: {self.parent.root_dir}\nblocks: {self.parent.blocks}\npositions: {self.parent.positions}\nbox: {self.parent.box}\ntransformations: {self.parent.transforms}\nout name: {self.parent.out_name}\n")

    def read_inputs_grafter(self, path_to_input_file: str):
        """Reads and validates input parameters from JSON file."""
        with open(path_to_input_file) as f:
            inputs = json.load(f)
        
        # Handle matrix configuration - store in parent system
        if inputs["matrix"]["file"]:
            self.parent.matrix = ["file", inputs["matrix"]["file"]]  # Changed to parent
        else:
            # Convert size to tuple and validate
            size = inputs["matrix"]["size"]
            if len(size) != 3:
                raise ValueError("Matrix size must have 3 dimensions [nx, ny, nz]")
            self.parent.matrix = ["build", tuple(size)]  # Changed to parent
        
        # Set other parameters - decide which should be in parent vs local
        self.parent.root_dir = Path(inputs["root directory"])  # Changed to parent
        self.parent.surf_dist = inputs["surface distance"] * 10  # nm to Å
        self.parent.grafting_density = inputs["grafting density"]
        self.parent.dispersity = [
            "mono" if inputs["chain dispersity"]["monodisperse"] else "poly",
            inputs["chain dispersity"]["monodisperse"] or inputs["chain dispersity"]["polydisperse"]
        ]
        self.parent.surf_geometry = "cylindrical" if inputs["surface geometry"]["cylindrical"] else "flat"
        self.parent.name_mapping = inputs["atom names"]
        self.parent.tilt_angle = inputs["perturbation"]
        self.parent.system_name = inputs["system name"]
        
        # Update grafting_params in parent
        self.parent.grafting_params = [
            self.parent.surf_dist,
            self.parent.grafting_density,
            self.parent.matrix,
            self.parent.dispersity,
            self.parent.surf_geometry,
            self.parent.name_mapping,
            self.parent.tilt_angle
        ]

        print(f"\nname: {self.parent.system_name}\nfolder: {self.parent.root_dir}\nsurface distance: {self.parent.surf_dist*.1} nm\ngrafting density: {self.parent.grafting_density} gps/nm^3\nmatrix: {self.parent.matrix}\ndispersity: {self.parent.dispersity}\nsurface geometry: {self.parent.surf_geometry}\natom names: {self.parent.name_mapping}\ntilt molecule: {self.parent.tilt_angle}\n")
        
        # If loading from file, create universe in parent
        if self.parent.matrix[0] == "file":
            self.parent.universe = mda.Universe(self.parent.matrix[1])
            self.parent.dataframes["bulk"] = pd.DataFrame(
                self.parent.universe.atoms.positions, 
                columns=["x", "y", "z"]
            )


    def write_xyz(self,fname,type,df,nTot,mode="w"):
        fout = open(fname,mode)
        if mode == "w":
            fout.write("%d\n" % (nTot))
            fout.write("Atoms. Timestep: 0\n")
        for i,row in df.iterrows():
            fout.write("%1d %5f %5f %5f\n" % (type, row["x"], row["y"], row["z"]))
        fout.close()
        return

    def write_atoms(self,fname,label=None,mode="w",id=[],atomTp=0,rsdNm=0,atomNm=0,q=0,m=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        for i in id:
            fout.write("%5d%8s%8d%8s%8s%6d%9.1f%9.1f\n" % (i,atomTp,i,rsdNm,atomNm,i,q,m))
        fout.close()
        return

    def write_bonds(self,fname,label=None,mode="w",id=[],f=0,l_eq=0,k_eq=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        for i in id[:-1]:
            fout.write("%5d%6d%6d%10.3f%10.3f\n" % (i,i+1,f,l_eq,k_eq))
        fout.close()
        return

    def write_angles(self,fname,label=None,mode="w",id=[],f=0,angle=0,k_eq=0):
        fout = open(fname,mode)
        if label != None:
            fout.write("%s\n" % label)
        fout.write("%5d%6d%6d%6d%10.3f%10.3f\n" % (id[0],id[1],id[2],f,angle,k_eq))
        fout.close()
        return

    def write_topol(self,fname,sysname,mols,includes,mode):
        fout = open(fname,mode)
        for i in includes:
            fout.write("%s\n" % i)
        if sysname:
            fout.write("\n[ system ]\n%s" % sysname)
            fout.write("\n\n[ molecules ]\n")
        if mols:
            for i in mols:      
                fout.write("%5s%10d\n" % (i[0],i[1]))
        fout.close()
        return