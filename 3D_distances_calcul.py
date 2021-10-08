import pandas as pd
import numpy as np
import math
import sqlite3
from scipy.spatial import distance_matrix

#https://noble.gs.washington.edu/proj/yeast-architecture/figures/3dfigures/3dmodel.pdb
atoms_coordinates = pd.read_csv('3dmodel.csv',
                          names = ("Atom","Atom_nb", "O", "EDG","chrom", "x", "y", "z", "1", "75","none"),
                          sep=',')

atoms_coordinates = atoms_coordinates.drop(["O","EDG","1","75","Atom_nb","Atom","none"],axis = 1)

atoms_coordinates = atoms_coordinates.set_index([pd.Index(list(range(1, 26539)))])

atoms_coordinates["chrom"] = atoms_coordinates["chrom"].replace(["A","B","C","D","E","F","G","H",
                                                                 "J","I","K","L","N","M","O","P"],
                                                                [1,2,3,4,5,6,7,8,
                                                                 9,10,11,12,13,14,15,16])

atoms_coordinates["chrom"] = pd.to_numeric(atoms_coordinates.chrom)

def format_segments_dataframe(chrom_number, atoms_coordinates):
    segments_coordinates = calcul_segments_coordinates(chrom_number, atoms_coordinates)

    segments_sizes = segments_coordinates.apply(calcul_segment_size, axis = 1)
    sum_segments_sizes = sum(segments_sizes)

    chrom_lenght = get_chromosome_lenght(chrom_number)

    sizes_on_chromosome = segments_sizes.apply(calcul_size_on_chrom, args = [chrom_lenght, sum_segments_sizes])

    start_bp = [0] + list(sizes_on_chromosome.cumsum().iloc[:len(sizes_on_chromosome) - 1])

    segments = pd.DataFrame(data = {"Segment_ID": sizes_on_chromosome.index,
                                              "start_bp": start_bp,
                                              "stop_bp": sizes_on_chromosome.cumsum()})

    return segments

def calcul_segments_coordinates(chrom_number, atoms_coordinates):
    segment_start = atoms_coordinates[atoms_coordinates["chrom"] == chrom_number]
    segment_start.index = range(1, len(segment_start) + 1)
    segment_start = segment_start.drop([len(segment_start)], axis = 0)
    segment_start.index = range(1, len(segment_start) + 1)

    segment_stop = atoms_coordinates[atoms_coordinates['chrom'] == chrom_number]
    segment_start.index = range(1, len(segment_start) + 1)
    segment_stop = segment_stop.iloc[1:]
    segment_stop.index = range(1, len(segment_stop) + 1)

    segments_coordinates = segment_start.merge(segment_stop, left_index = True,
                                               right_index = True,
                                               suffixes = ["_start", "_stop"] )
    segments_coordinates = segments_coordinates.drop(["chrom_start","chrom_stop"], axis = 1)

    return segments_coordinates

def calcul_segment_size(segments_coordinates):

    return math.sqrt((segments_coordinates.x_stop - segments_coordinates.x_start) ** 2 +
                     (segments_coordinates.y_stop - segments_coordinates.y_start) ** 2 +
                     (segments_coordinates.z_stop - segments_coordinates.z_start) ** 2)

def get_chromosome_lenght(chrom_number):
    #SQL request
    db_connexion = sqlite3.connect('../SCERE.db')

    cursor = db_connexion.cursor()

    chromosome_length = cursor.execute("""
    SELECT length
    FROM chromosome_length
    """)

    chromosome_length = chromosome_length.fetchall()
    chromosome_length = pd.DataFrame(chromosome_length, columns = ["length"], index = list(range(1,18)))

    return chromosome_length.loc[chrom_number][0]

def calcul_size_on_chrom(segments_sizes, chrom_lenght, sum_segments_sizes):

    return (segments_sizes * chrom_lenght) / sum_segments_sizes

# ###Segments and locus connexions.

def connect(segments, loci):
    loci = loci[loci.Feature_type != 'CDS']
    loci = loci[loci.Start_coordinate != ""]
    loci.index = range(1, len(loci) + 1)

    intersections = segments.apply(get_intersection, args = [loci], axis = "columns")

    return intersections

def get_intersection(segment, loci):

    on = loci.assign(start_bp = segment.start_bp)
    on = on.assign(stop_bp = segment.stop_bp)

    conditions = [(on.Start_coordinate <= on.start_bp) & (on.start_bp <= on.Stop_coordinate),
                  (on.start_bp <= on.Start_coordinate) & (on.Start_coordinate <= on.stop_bp)]

    choices = [True, True]

    on = on.assign(on_segment = np.select(conditions, choices, default = False))

    locus_on_segment = on.Primary_SGDID[on.on_segment == True]
    locus_on_segment.index = range(1, len(locus_on_segment) + 1)

    return locus_on_segment

def string_to_list(string):
    return list(string.split(", "))

def get_chrom_info(database, chrom_number, info, source, condition, group_by, order_by):
    #SQL request
    db_connexion = sqlite3.connect(database)

    cursor = db_connexion.cursor()

    chrom = cursor.execute("""
    SELECT """ + info + """
    FROM """ + source + """
    WHERE Chromosome == """ + chrom_number + condition + group_by + """
    ORDER BY """ + order_by)


    list_info = string_to_list(info)

    #pandas dataframe formatting
    chrom = chrom.fetchall()
    chrom = pd.DataFrame(chrom, columns = list_info)

    return chrom

def format_atoms_coordinates(new_segments):

    plotly_segments = pd.DataFrame(columns = ["x", "y", "z", "Primary_SGDID"])

    for i in range(0, len(new_segments)):
        plotly_segments = plotly_segments.append({"x": new_segments["x_start"][i],
                                                  "y": new_segments["y_start"][i],
                                                  "z": new_segments["z_start"][i],
                                                  "Primary_SGDID": new_segments["Primary_SGDID"][i]},
                                                  ignore_index=True)

        plotly_segments = plotly_segments.append({"x": new_segments["x_stop"][i],
                                                  "y": new_segments["y_stop"][i],
                                                  "z": new_segments["z_stop"][i],
                                                  "Primary_SGDID": new_segments["Primary_SGDID"][i]},
                                                  ignore_index=True)

        plotly_segments = plotly_segments.append({"x": None,
                                                  "y": None,
                                                  "z": None,
                                                  "Primary_SGDID": new_segments["Primary_SGDID"][i]},
                                                  ignore_index=True)

    return plotly_segments

print(1)
loci_series_list = []

for c in range(1, 17):

    chrom_number = str(c)

    loci = get_chrom_info('../SCERE.db',
                          chrom_number,
                          """Primary_SGDID, Start_coordinate, Stop_coordinate, Feature_type""",
                          "SGD_Features",
                          """""",
                          """""",
                          """ Start_coordinate""")

    segments = format_segments_dataframe(c, atoms_coordinates)

    loci_series_list = loci_series_list + [connect(segments, loci)]

segments_loci = pd.concat(loci_series_list)
segments_loci.index = range(1, len(segments_loci) + 1)

list_segments_loci = segments_loci

for i in range(1, len(list_segments_loci) + 1):
    list_segments_loci.at[i, 1] = list(list_segments_loci.loc[i].dropna())

list_segments_loci.index = range(1, len(list_segments_loci) + 1)
list_segments_loci = list_segments_loci[1]
list_segments_loci = list(list_segments_loci)

#Récupération des coordonnées des 26522 segments 3D
segments_coordinates = calcul_segments_coordinates(1, atoms_coordinates)

for c in range(2, 17):

    chrom_segments_coordinates = calcul_segments_coordinates(c, atoms_coordinates)

    segments_coordinates = segments_coordinates.append(chrom_segments_coordinates)

# segments_coordinates.index = range(0, len(segments_coordinates))
segments_coordinates = segments_coordinates.reset_index()
print(2)
#new_segments stocke dans l'ordre les segments, dont ceux qui sont subdivisés
new_segments = pd.DataFrame(columns = ["x_start", "y_start", "z_start", "x_stop", "y_stop", "z_stop", "Primary_SGDID"])

#loci_3Dloc stocke les loci et leur position dans l'espace (le start du segments associé au loci)

#on parcourt la liste des segments, avec leurs loci associés
for s in range(0, len(list_segments_loci)):

    #liste des loci qui n'ont pas encore été associés à un segment
    new_loci = [x for x in list_segments_loci[s] if x not in list(new_segments["Primary_SGDID"].values)]

    #dans le cas où il y a plus d'un loci associé au segment, on le subdivise par le nombre de loci.
    if len(new_loci) > 0:
        segment_vect_x = segments_coordinates.loc[s, "x_stop"] - segments_coordinates.loc[s, "x_start"]
        segment_vect_y = segments_coordinates.loc[s, "y_stop"] - segments_coordinates.loc[s, "y_start"]
        segment_vect_z = segments_coordinates.loc[s, "z_stop"] - segments_coordinates.loc[s, "z_start"]
        for new_loci_idx in range(0, len(new_loci)):
            # New start point
            x_start = segments_coordinates.loc[s, "x_start"] + (new_loci_idx/len(new_loci)) * segment_vect_x
            y_start = segments_coordinates.loc[s, "y_start"] + (new_loci_idx/len(new_loci)) * segment_vect_y
            z_start = segments_coordinates.loc[s, "z_start"] + (new_loci_idx/len(new_loci)) * segment_vect_z
            # New stop point
            x_stop = segments_coordinates.loc[s, "x_start"] + ((new_loci_idx+1)/len(new_loci)) * segment_vect_x
            y_stop = segments_coordinates.loc[s, "y_start"] + ((new_loci_idx+1)/len(new_loci)) * segment_vect_y
            z_stop = segments_coordinates.loc[s, "z_start"] + ((new_loci_idx+1)/len(new_loci)) * segment_vect_z
            # Store new segment
            new_segments = new_segments.append({"Primary_SGDID": new_loci[new_loci_idx],
                                                "x_start": x_start,
                                                "y_start": y_start,
                                                "z_start": z_start,
                                                "x_stop": x_stop,
                                                "y_stop": y_stop,
                                                "z_stop": z_stop},
                                                ignore_index=True)

    else:
        new_segments = new_segments.append({"Primary_SGDID": None,
                                        "x_start": segments_coordinates.loc[s, "x_start"],
                                        "y_start": segments_coordinates.loc[s, "y_start"],
                                        "z_start": segments_coordinates.loc[s, "z_start"],
                                        "x_stop": segments_coordinates.loc[s, "x_stop"],
                                        "y_stop": segments_coordinates.loc[s, "y_stop"],
                                        "z_stop": segments_coordinates.loc[s, "z_stop"]},
                                        ignore_index=True)

loci_3Dloc = new_segments.drop(["x_stop", "y_stop", "z_stop"], axis = 1)
loci_3Dloc = loci_3Dloc[- loci_3Dloc["Primary_SGDID"].isna()]

loci_3Dloc = loci_3Dloc.sort_values("Primary_SGDID")
loci_3Dloc.index = range(1, len(loci_3Dloc) + 1)

Primary_SGDID = loci_3Dloc["Primary_SGDID"]
loci_3Dloc = loci_3Dloc.drop(["Primary_SGDID"], axis=1)
loci_3Dloc = loci_3Dloc.to_numpy()

print(3)
distances = pd.DataFrame(distance_matrix(loci_3Dloc, loci_3Dloc))

distances.index = Primary_SGDID
distances.columns = Primary_SGDID
distances.columns.name = "Primary_SGDID_bis"
distances.values[np.triu_indices_from(distances, 0)] = np.nan

edges_list = distances.stack().dropna().reset_index()
edges_list.rename(columns = {0: "3D_distances"}, inplace = True)
edges_list = edges_list.sort_values(by="Primary_SGDID")
edges_list.to_parquet('3D_distances.parquet.gzip', engine = "pyarrow")

plotly_segments = format_atoms_coordinates(new_segments)
plotly_segments.to_csv("plotly_segments.csv", index = False, columns = ["x", "y", "z", "Primary_SGDID"])