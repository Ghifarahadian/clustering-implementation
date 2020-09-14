# Databricks notebook source
# IMPORTANT: Change this to specify running in local laptop or databricks
DATABRICKS = False

# COMMAND ----------

import os, re, sys, math, random, time
from pyspark import SparkConf, SparkContext

# COMMAND ----------

# Python version: 3.7.6
# Package used: pyspark, with spark version 3.0.0-preview2

# important file directory
DBFS_DIR = ""
if DATABRICKS:
    DBFS_DIR = "/Lab 2/"
RDD_DIR = DBFS_DIR + "input/"

# Parameters. All adjustable variables are declared here
K_CLUSTERS = 2  # num of clusters
N_UPDATE_THRESHOLD = 20  # maximum run for updating centroids
update_cs = True  # set to True to update the centroid using cosine similarity
update_ed = True  # set to True to update the centroid using euclidean distance
#random.seed(69)  # uncomment for reproducibility

# COMMAND ----------

"""
Datapoint class. It links to its clusters, both from cosine similarity and euclidean distance
Its point coordinate is in RDD
"""
class DataPoint:
	def __init__(self, name, rdd):
		self.name = name
		self.cluster_cs = None  # points to its cosine similarity cluster
		self.cluster_ed = None  # points to its euclidean distance cluster
		self.rdd = rdd

"""
Cluster class. It also keeps tracks on the points in its clusters, both from cosine and euclidean
Its centroid is in RDD
"""
class Cluster:
	def __init__(self, name, ctrd):
		self.name = name
		self.ctrd = ctrd  # centroid
		self.points = []  # list of all points belonging to this cluster

# to keep track of datapoints and clusters
datapoints = []
clusters_cs = []
clusters_ed = []

# initiate the spark
conf = SparkConf()
sc = SparkContext(conf=conf)

# COMMAND ----------

# read RDDs for each points that are saved from Task A
if DATABRICKS:
    rdds = [f.name[:-1] for f in dbutils.fs.ls(RDD_DIR)]
else:
    rdds = os.listdir(RDD_DIR)
rdds.remove("docf")  # we don't load docf here
#rdds = ["f11", "f12", "f13", "f31", "f32", "f33"]  # test data for lightweight purpose
for rdd_name in rdds:
	rdd_new = sc.sequenceFile(RDD_DIR + rdd_name)
	rdd_new.persist()
	datapoints.append(DataPoint(name=rdd_name, rdd=rdd_new))

# read docf RDD. It contains all the words i.e. all the dimensions. The value does not matter
docf = sc.sequenceFile(RDD_DIR + "docf")

"""
initialize the K-cluster
to initialize, I choose random point from the input as the centroid of the cluster
both clusters from cosine similarity and euclidean distance are generated here, so the clusters
from two distance functions can be computed simultaneously
"""
for i in range(0, K_CLUSTERS):
	for j in range(0, 2):
		rnd = random.randint(1, len(rdds)-1)
		rdd_new = sc.sequenceFile(RDD_DIR + "f" + str(rnd))
		rdd_new.persist()
		if j==0:
			clusters_cs.append(Cluster(name=i, ctrd=rdd_new))
		elif j==1:
			clusters_ed.append(Cluster(name=i, ctrd=rdd_new))

# COMMAND ----------

"""
for each data point, place it in the cluster whose current centroid it is nearest
both cosine similarity and euclidean distance are implemented here. Variable update_cs and update_ed governs whether
the cosine similarity/euclidean distance is updated
the cosine similarity is implemented by doing element multiplication then summing the results, similar way
as task A
the euclidean distance is computed by taking delta for each dimension, square them, sum them, and take the 
square root
"""
print("Step 1")
for p in datapoints:
	# cosine similarity
    if update_cs:
        cs = 0
        for c in clusters_cs:
            pc_rdd = p.rdd.join(c.ctrd)
            pc_rdd.persist()
            elem_mul = pc_rdd.map(lambda w: (w[0], w[1][0]*w[1][1]))
            elem_mul.persist()
            cs_temp = elem_mul.values().sum()
            if cs < cs_temp:
                cs = cs_temp
                p.cluster_cs = c
        p.cluster_cs.points.append(p)

	# euclidean distance
    if update_ed:
        ed = 1000000000000  # hopefully big enough lol
        for c in clusters_ed:
            pc_rdd = p.rdd.join(c.ctrd)
            pc_rdd.persist()
            elem_delta_sq = pc_rdd.map(lambda w: (w[0], (w[1][0]-w[1][1])*(w[1][0]-w[1][1])))
            elem_delta_sq.persist()
            ed_temp = math.sqrt(elem_delta_sq.values().sum())
            if ed > ed_temp:
                ed = ed_temp
                p.cluster_ed = c
        p.cluster_ed.points.append(p)

clusters_cs_str = ""
for c in clusters_cs:
    clusters_cs_str += str(c.name) + "\n"
    clusters_cs_str += str([p.name for p in c.points]) + "\n\n"
print(clusters_cs_str)

# COMMAND ----------

"""
Update the centroids by using the average of the points inside the cluster until convergence
or the update count exceeds threshold. The average is performed for each dimension
"""
count = 0
count_cs = 0
count_ed = 0
while update_cs or update_ed:
    # maximum update
    if not update_cs:
        count_ed += 1
    if not update_ed:
        count_cs += 1
    count += 1
    if count_ed >= N_UPDATE_THRESHOLD or count_cs >= N_UPDATE_THRESHOLD or count >= N_UPDATE_THRESHOLD:
        break

    # update the locations of centroids
    print("Step 2")
    # cosine similarity
    if update_cs:
        for c in clusters_cs:
            total_points = len(c.points)
            if total_points == 0:
                continue
            cluster_rdd_all = sc.union([p.rdd for p in c.points])
            cluster_rdd_all.persist()
            cluster_sum_rdd = cluster_rdd_all.reduceByKey(lambda l1, l2: l1 + l2)
            cluster_sum_rdd.persist()
            ctr_new = cluster_sum_rdd.map(lambda l: (l[0], l[1]/total_points))
            ctr_new.persist()
            c.ctrd = ctr_new
            c.points = []

    # euclidean distance
    if update_ed:
        for c in clusters_ed:
            total_points = len(c.points)
            if total_points == 0:
                continue
            cluster_rdd_all = sc.union([p.rdd for p in c.points])
            cluster_rdd_all.persist()
            cluster_sum_rdd = cluster_rdd_all.reduceByKey(lambda l1, l2: l1 + l2)
            cluster_sum_rdd.persist()
            ctr_new = cluster_sum_rdd.map(lambda l: (l[0], l[1]/total_points))
            ctr_new.persist()
            c.ctrd = ctr_new
            c.points = []

    # reassign points to closest centroid
    print("Step 3")
    update_cs_temp = False  # to keep track whether the cluster has converged
    update_ed_temp = False  # to keep track whether the cluster has converged
    for p in datapoints:
        # cosine similarity
        if update_cs:
            cs = 0
            new_cluster_cs = p.cluster_cs
            for c in clusters_cs:
                pc_rdd = p.rdd.join(c.ctrd)
                pc_rdd.persist()
                elem_mul = pc_rdd.map(lambda w: (w[0], w[1][0]*w[1][1]))
                elem_mul.persist()
                cs_temp = elem_mul.values().sum()

                if cs < cs_temp:
                    cs = cs_temp
                    new_cluster_cs = c

            if p.cluster_cs != new_cluster_cs:
                p.cluster_cs = new_cluster_cs
                update_cs_temp = True
            p.cluster_cs.points.append(p)


        # euclidean distance
        if update_ed:
            ed = 1000000000000  # hopefully big enough lol
            new_cluster_ed = p.cluster_ed
            for c in clusters_ed:
                pc_rdd = p.rdd.join(c.ctrd)
                pc_rdd.persist()
                elem_delta_sq = pc_rdd.map(lambda w: (w[0], (w[1][0]-w[1][1])*(w[1][0]-w[1][1])))
                elem_delta_sq.persist()
                ed_temp = math.sqrt(elem_delta_sq.values().sum())

                if ed > ed_temp:
                    ed = ed_temp
                    new_cluster_ed = c

            if p.cluster_ed != new_cluster_ed:
                p.cluster_ed = new_cluster_ed
                update_ed_temp = True
            p.cluster_ed.points.append(p)

    if not update_cs_temp:  # already finish updating
        update_cs = False
    if not update_ed_temp:  # already finish updating
        update_ed = False

    clusters_cs_str = ""
    for c in clusters_cs:
        clusters_cs_str += str(c.name) + "\n"
        clusters_cs_str += str([p.name for p in c.points]) + "\n\n"
    print("Current clusters:")
    print(clusters_cs_str)
        
    print("Update count: ", count)

# COMMAND ----------

# write the output. Pretty self-explanatory :)
if DATABRICKS:
    dbutils.fs.rm(dir=DBFS_DIR + "cluster_cs.txt")
    dbutils.fs.rm(dir=DBFS_DIR + "cluster_ed.txt")
    
    clusters_cs_str = ""
    for c in clusters_cs:
        clusters_cs_str += str(c.name) + "\n"
        clusters_cs_str += str([p.name for p in c.points]) + "\n\n"

    clusters_db_str = ""
    for c in clusters_ed:
        clusters_db_str += str(c.name) + "\n"
        clusters_db_str += str([p.name for p in c.points]) + "\n\n"
    
    dbutils.fs.put(file = DBFS_DIR + "cluster_cs.txt", contents = clusters_cs_str)
    dbutils.fs.put(file = DBFS_DIR + "cluster_ed.txt", contents = clusters_db_str)
else:
    os.system("rm cluster_cs.txt")
    os.system("rm cluster_ed.txt")
    f_cs = open("cluster_cs.txt", "a")
    f_ed = open("cluster_ed.txt", "a")

    for c in clusters_cs:
        f_cs.writelines(str(c.name) + "\n")
        f_cs.writelines(str([p.name for p in c.points]) + "\n\n")
    f_cs.close()

    for c in clusters_ed:
        f_ed.writelines(str(c.name) + "\n")
        f_ed.writelines(str([p.name for p in c.points]) + "\n\n")
    f_ed.close()

print("Completed. The output can be found in cluster_cs.txt and/or cluster_ed.txt file")

# COMMAND ----------

sc.stop()
