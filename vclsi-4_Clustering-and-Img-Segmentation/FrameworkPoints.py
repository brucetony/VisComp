import numpy as np
import scipy
from scipy import misc
from matplotlib import pyplot,cm

def GetPointEdges(Points,SigmaDistance,EdgeRadius):
    """This function constructs a graph describing similarity of points in the given 
       array.
      \param Points An array of shape (nPoint,2) where each row provides the 
             coordinates of one of nPoint points in the plane.
      \param SigmaDistance The standard deviation of the Gaussian distribution used 
             to weigh down longer edges.
      \param EdgeRadius A positive float providing the maximal length of edges.
      \return A tuple (EdgeWeight,EdgeIndices) where EdgeWeight is an array of 
              length nEdge providing the weight of all produced edges and 
              EdgeIndices is an integer array of shape (nEdge,2) where each row 
              provides the indices of two pixels which are connected by an edge."""
    


def GetLaplacian(nVertex,EdgeWeight,EdgeIndices):
    """Constructs a matrix providing the Laplacian for the given graph. 
      \param nVertex The number of vertices in the graph (resp. pixels in the 
             image).
      \param EdgeWeight A one-dimensional array of nEdge floats providing the weight 
             for each edge.
      \param EdgeIndices An integer array of shape (nEdge,2) where each row provides 
             the vertex indices for one edge.
      \return A matrix providing the Laplacian for the given graph."""
    


def GetFiedlerVector(Laplacian):
    """Given the Laplacian matrix of a graph this function computes the normalized 
       Eigenvector for its second-smallest Eigenvalue (the so-called Fiedler vector) 
       and returns it."""
    



if(__name__=="__main__"):
    # This list of points is to be clustered
    Points=np.asarray([(-8.097,10.680),(-3.902,8.421),(-9.711,7.372),(0.859,12.859),(4.732,11.084),(-0.594,9.147),(-4.224,13.585),(-9.066,11.891),(-13.181,8.663),(-12.374,3.983),(-11.406,-2.068),(-9.630,2.854),(-13.665,-6.667),(-15.521,-0.454),(-15.117,-6.587),(-11.970,-10.621),(-6.000,-12.799),(-2.853,-14.978),(-8.501,-10.217),(2.311,-11.670),(3.441,-14.171),(5.861,-10.137),(10.138,-6.909),(15.382,-5.215),(14.091,0.675),(11.187,3.903),(8.685,8.502),(7.879,11.649),(5.216,10.680),(11.025,6.888),(13.446,2.612),(12.962,-7.393),(8.363,-9.330),(-0.594,-0.212),(1.666,1.401),(1.424,-1.019),(-0.351,-2.552),(-2.127,0.675),(-0.271,2.128),(-4.743,-4.016)]);
    nVertex=Points.shape[0];
    
    # Construct the graph for the points
    EdgeWeight,EdgeIndices=GetPointEdges(Points,1.8,7.0);
    # Construct the Laplacian matrix for the graph
    Laplacian=GetLaplacian(nVertex,EdgeWeight,EdgeIndices);
    # Compute the Fiedler vector
    FiedlerVector=GetFiedlerVector(Laplacian);
    
    # Show the results
