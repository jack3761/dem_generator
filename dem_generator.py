import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score


"""
    Extracts contours from an image using a specified colour space and threshold.

    Args:
        image (numpy.ndarray): The input image to extract contours from.
        colour_space_in (str): The colour space to use for contour extraction. Possible
            values are "Grayscale", "Hue", "Saturation", or "Value".
        threshold (int or None): The threshold value for contour extraction. If None,
            Otsu's thresholding method is used.

    Returns:
        list of numpy.ndarray: A list of numpy arrays representing the contours
        found in the image.
"""
def extractContours(image, colour_space_in, threshold):

    # Convert image to different colour channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Assign colour space from user input
    if colour_space_in == "Grayscale":
        colour_space = gray
    elif colour_space_in == "Hue":
        colour_space = h
    elif colour_space_in == "Saturation":
        colour_space = s
    elif colour_space_in == "Value":
        colour_space = v

    # Threshold image using given threshold method
    if threshold == None:
        _, binary = cv2.threshold(colour_space, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(colour_space, threshold, 255, cv2.THRESH_BINARY_INV)

    # Extract contours from image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Only take one border for each line
    contours = [contours[x] for x in range(len(contours)) if x%2 == 0]

    return contours

"""
    Draws each contour in each cluster on a black background with a different color for each
    cluster. If show_labels is True, it also displays the cluster number next to the bottom
    contour in the cluster.

    Args:
        image (numpy.ndarray): The input image to draw the clusters on.
        clusters (list of list of numpy.ndarray): A list of lists of numpy arrays representing
            the clusters to be drawn on the image.
        show_labels (bool): If True, displays the cluster number next to the bottom contour in
            each cluster.

    Returns:
        numpy.ndarray: The resulting image with the drawn clusters.
    """
def draw_clusters(image, clusters, show_labels):
    # Create black grid to draw on
    background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Iterate through each contour in each cluster 
    for i, contours in enumerate(clusters):

        # Assign each cluster a different colour
        color = np.random.randint(0, 255, size=(3,))
        color = tuple(map(int, color))

        # Draw each contour in the cluster
        for j, contour in enumerate(contours):
            cv2.drawContours(background, [contour], -1, color, thickness=2)
            last_contour = contour

        # Draw the cluster number label next to the bottom contour in the cluster
        if show_labels:
            rightmost = tuple(last_contour[last_contour[:, :, 0].argmax()][0])
            cv2.putText(background, str(i), (rightmost[0]+10, rightmost[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display image
    cv2.imshow('Clustered Contours', background)
    return background


"""
    Extracts features from a list of contours.

    Args:
        contours (list of numpy.ndarray): A list of numpy arrays representing the contours to
            extract features from.

    Returns:
        numpy.ndarray: A 2D numpy array containing the features extracted from the contours.
            Each row represents a contour, and the columns contain the area, perimeter,
            aspect ratio, centroid x-coordinate, and centroid y-coordinate of each contour.
"""
def get_features(contours):

    # Create an empty list to store the features extracted from each contour
    features = []

    # Loop through each contour
    for contour in contours:
        
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Get the coordinates, width and height of the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the aspect ratio of the bounding rectangle
        aspect_ratio = float(w) / h
        
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # Calculate the centroid coordinates of the contour
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        
        # Append the features of the current contour to the features list
        features.append([area, perimeter, aspect_ratio, centroid_x, centroid_y])

    features = np.array(features)

    return features


"""
    Normalizes the features by subtracting the mean and dividing by the standard deviation.

    Args:
        features (numpy.ndarray): A 2D numpy array containing the features to normalize. Each row
            represents a contour, and the columns contain the area, perimeter, aspect ratio,
            centroid x-coordinate, and centroid y-coordinate of each contour.

    Returns:
        numpy.ndarray: A 2D numpy array containing the normalized features. Each row represents a
            contour, and the columns contain the area, perimeter, aspect ratio, centroid
            x-coordinate, and centroid y-coordinate of each contour.
"""
def normalise_features(features):

    # calculated the normalised features with the mean and standard deviation
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalised_features = (features - mean) / std

    return normalised_features

"""
    Determines the optimal number of clusters to use for KMeans clustering based on the 
    silhouette scores of different number of clusters.

    Args:
        features (numpy.ndarray): An array containing the features extracted from contours.
        show_plot (bool): If True, shows a plot of the silhouette scores for each number of clusters.

    Returns:
        int: The optimal number of clusters based on the silhouette scores.
    """

def optimal_kmeans_clusters(features, show_plot=False):
    # find the silhouette scores for different k clusters

    sil_scores = []
    for k in range(2, len(features)):
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(features)
        sil_score = silhouette_score(features, labels)
        if show_plot:
            print(str(k) + " clusters: " + str(sil_score))
        sil_scores.append(sil_score)
    
    # Choose the optimal number of clusters based on the silhouette score
    optimal_k = np.argmax(sil_scores) + 2

    return optimal_k


"""
    Clusters a set of contours using a specified clustering method.
    
    Args:
    - contours (list): a list of contours.
    - method (str): the clustering method to use. Can be either 'Hierarchical' or 'KMeans'.
    - n_clusters (int): the number of clusters to form. Only used for KMeans clustering.
    - linkage_method (str): the linkage method to use for Hierarchical clustering. 
                            Only used for Hierarchical clustering. Default is 'ward'.
                            
    Returns:
    - clustered_contours (list): a list of lists containing the contours for each cluster.
    - n_clusters (int): the number of clusters formed.
    - Z (ndarray): the linkage matrix. Only returned if method is 'Hierarchical'.
    """

def find_clusters(contours, method, n_clusters=None, linkage_method="ward"):
    features = []

    # Find the normalised features for the set of contours
    features = get_features(contours)
    normalised_features = normalise_features(features)

    # Perform chosen method of clustering
    if method == "Hierarchical":

        # Use provided linkage method to create a linkage matrix
        Z = linkage(normalised_features, method=linkage_method)

        # Find elbow of the dendrogram for number of clusters
        last = Z[-10:, 2]
        last_rev = last[::-1]
        idx = np.arange(1, len(last) + 1)
        slope = np.diff(last_rev)
        elbow = np.argmin(slope) + 1

        # Perform hierarchical clustering with the number of clusters determined automatically
        if n_clusters == None:
            n_clusters = idx[elbow]

        # Find the labels from the linkage matrix
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

    else:
        Z = None
        if n_clusters != None:

            # Perform k-means clustering with the specified number of clusters
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            labels = kmeans.fit_predict(normalised_features)

        else:
            # Find optimal k value for clusters
            optimal_k = optimal_kmeans_clusters(normalised_features, False)
            n_clusters = optimal_k

            # Perform k-means clutsering with the suggested number of clusters
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            labels = kmeans.fit_predict(normalised_features)

    # Generate clustered_contours array from the contours and the labels
    clustered_contours = [[] for i in range(n_clusters)]
    for i, contour in enumerate(contours):
        cluster = labels[i]
        clustered_contours[cluster].append(contour)

    # If there is more than one cluster then show their metrics
    if n_clusters > 1:

        # Calcluate the solhouette score for the clustered contours
        silhouette_coef = silhouette_score(normalised_features, labels)

        # Calculate the Calinski Harabasz index for the clustered contours
        calinski_harabasz = calinski_harabasz_score(normalised_features, labels)

        print(method)
        print("N Contours: " + str(len(contours)))
        print("N_clusters: " + str(n_clusters))
        print("Silhouette score: " + str(silhouette_coef))
        print("Calinski Harabaskz index: " + str(calinski_harabasz) + "\n")

    return clustered_contours, n_clusters, Z

"""
    Plots contours of clusters and generates a surface plot using the provided image and clusters.

    Args:
    - image: 2D numpy array, the image to plot
    - clusters: list of lists, a list of lists of contours that belong to each cluster
    - contour_interval: int or float, the interval between contour lines
    - ax: matplotlib axis, the axis to plot on
    - trisurf: bool, optional, whether to use a trisurf plot instead of a surface plot, default is False
    - show_contours: bool, optional, whether to show the individual contour lines, default is False
    - show_surface: bool, optional, whether to show the surface plot, default is True

    Returns:
    - ax: matplotlib axis, the axis with the plotted data
    """
def plot_clusters(image, clusters, contour_interval, ax, trisurf=False, show_contours=False, show_surface=True):
    z = [[]]
    x = []
    y = []
    contour_points = []
    elevations = []

    # Generate meshgrid with the dimensions of the provided image to use for the surface plot
    xi = np.linspace(0, image.shape[0], image.shape[0] + 1)
    yi = np.linspace(0, image.shape[1], image.shape[1] + 1)
    X, Y = np.meshgrid(xi, yi)
    Z = np.zeros((len(xi), len(yi)))
    Z = np.transpose(Z)

    # Set axis limits so there is more context for size in the plot
    ax.set_zlim(0, 100)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])

    # Iterate through each cluster
    for i, cluster in enumerate(clusters):
        z = [[]]
        x = []
        y = []
        
        # Find the start elevation to be able to calculate contour elevations
        start_elevation = find_start_elevation(clusters, cluster[len(cluster)-1]) * contour_interval

        # Calculate elevations for the contours using the given contour interval
        contour_elevations = [(x * contour_interval + start_elevation) for x in range(len(cluster))]
        
        # Iterate through each contour in the cluster
        for j, contour in enumerate(cluster, start=1):

            # Plot the contour line
            if show_contours:
                x.append(contour[:,:,0].flatten())
                y.append(contour[:,:,1].flatten())
                z = contour_elevations[-j]
                ax.plot(y[j-1], x[j-1], z)

            # Append the elevation to array
            elevation = contour_elevations[-j]
            elevations.append([elevation] * len(contour))

            # Append contour points to array
            contour_points.append(contour.squeeze())

    contour_points = np.concatenate(contour_points)
    elevations = np.concatenate(elevations)

    # Interpolate elevations for all points on the meshgrid
    Z = griddata(contour_points, elevations, (Y, X), method='linear')

    # Generate surface plot with given method
    if show_surface:
        if trisurf == False: 
            plt.colorbar(ax.plot_surface(X, Y, Z, cmap='plasma'))
        else:
            x_flat = X.ravel()
            y_flat = Y.ravel()
            plt.colorbar(ax.plot_trisurf(x_flat, y_flat, Z.ravel(), cmap='plasma'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


"""
    Finds the starting elevation for the contours to be plotted

    Args:
    - clusters: list of lists, a list of lists of contours that belong to each cluster
    - start_contour - the starting contour to compare if it is contained by another line

    Returns:
    - start_elevation - starting value to calculate elevations from
"""
def find_start_elevation(clusters, start_contour):
    start_elevation = 0

    # Squeeze contour to 1D array
    contourPoints = np.squeeze(np.array(start_contour)).astype(int)

    # Iterate through other clusters
    for i, cluster in enumerate(clusters):
        for j, contour in enumerate(cluster):
            # Only apply if other contour is larger
            if len(np.squeeze(np.array(contour))) < len(contourPoints):
                continue
            else:
                # Check if the provided contour is contained by the other
                if is_inside_contour(contourPoints, contour):
                    start_elevation += 1

    return start_elevation

"""
    Determines whether a contour is contained by another one

    Args:
    - inner_contour - the contour to be compared to see if it is contained
    - outer_contour - the contour to see if it's containg the other

    Returns:
    - bool
"""
def is_inside_contour(inner_contour, outer_contour):

    # Convert arrays for point polygon test
    outer_points = np.array(outer_contour)
    inner_points = np.squeeze(np.array(inner_contour)).astype(int)

    # Find distances between points to find if the contour is contained by the given one
    distances = [cv2.pointPolygonTest(outer_points, (int(point[0]), int(point[1])), False) for point in inner_points]

    # Check if the contour is contained
    if all(distance > 0 for distance in distances):
        return True
    else:
        return False
    

class MyGUI:

    def __init__(self, master):
        self.master = master

        self.extract_frame = tk.Frame(master)
        self.extract_frame.pack()

        # Add a dropdown menu for the color map
        self.color_label = tk.Label(self.extract_frame, text="Color Map:")
        self.color_label.pack(side="left")
        self.color_options = ["Grayscale", "Hue", "Saturation", "Value"]
        self.color_var = tk.StringVar(value="Grayscale")
        self.color_dropdown = tk.OptionMenu(self.extract_frame, self.color_var, *self.color_options)
        self.color_dropdown.pack(side="left")

        # Add a checkbox for the threshold type
        self.threshold_var = tk.BooleanVar(value=True)
        self.threshold_check = tk.Checkbutton(self.extract_frame, text="Automatic Threshold", variable=self.threshold_var)
        self.threshold_check.pack(side="left")

        # Add a text input for the manual threshold
        self.manual_label = tk.Label(self.extract_frame, text="Manual Threshold:")
        self.manual_label.pack(side="left")
        self.manual_entry = tk.Entry(self.extract_frame, width=5)
        self.manual_entry.pack(side="left")

        # Add a checkbox for the threshold type
        self.show_clusters_var = tk.BooleanVar(value=True)
        self.show_clusters_check = tk.Checkbutton(self.extract_frame, text="Show Clusters", variable=self.show_clusters_var)
        self.show_clusters_check.pack(side="left")

        # Add a checkbox to show contour labels
        self.show_cluster_labels_var = tk.BooleanVar(value=True)
        self.show_cluster_labels_check = tk.Checkbutton(self.extract_frame, text="Show Labels", variable=self.show_cluster_labels_var)
        self.show_cluster_labels_check.pack(side="left")

        # Add a checkbox for the showing the contours
        self.show_contours_var = tk.BooleanVar(value=True)
        self.show_contours_check = tk.Checkbutton(self.extract_frame, text="Show Contours on Plot", variable=self.show_contours_var)
        self.show_contours_check.pack(side="left")

        # Add a checkbox for the showing the surface
        self.show_surface_var = tk.BooleanVar(value=True)
        self.show_surface_check = tk.Checkbutton(self.extract_frame, text="Show Surface on Plot", variable=self.show_surface_var)
        self.show_surface_check.pack(side="left")

        # Create a Frame to hold the UI elements
        self.model_frame = tk.Frame(master)
        self.model_frame.pack()

        # Add a label and a text entry for the input file name
        self.file_label = tk.Label(self.model_frame, text="Input File Name:")
        self.file_label.pack(side="left")
        self.file_entry = tk.Entry(self.model_frame)
        self.file_entry.pack(side="left")

        # Add options for the type of surface model
        self.surface_label = tk.Label(self.model_frame, text="Surface Model:")
        self.surface_label.pack(side="left")
        self.surface_options = ["Grid Surface", "Triangulate Surface"]
        self.surface_var = tk.StringVar(value="Grid Surface")
        self.surface_dropdown = tk.OptionMenu(self.model_frame, self.surface_var, *self.surface_options)
        self.surface_dropdown.pack(side="left")

        # Add spinbox to provide the contour interval
        self.spinbox_label = tk.Label(self.model_frame, text="Contour Interval")
        self.spinbox_label.pack(side="left")
        self.spinbox_var = tk.IntVar(value=5)
        self.spinbox = tk.Spinbox(self.model_frame, from_=0, to=100, increment=5, textvariable=self.spinbox_var, width=3)
        self.spinbox.pack(side="left")

        # Add a n_clusters for the number of clusters
        self.n_clusters_label = tk.Label(self.model_frame, text="No. of Clusters:")
        self.n_clusters_label.pack(side="left")
        self.n_clusters_var = tk.IntVar(value=1)
        self.n_clusters_entry = tk.Spinbox(self.model_frame, from_=0, to=100, increment=1, textvariable=self.n_clusters_var, width=3)
        self.n_clusters_entry.pack(side="left")

        # Add a button to submit the variables
        self.submit_button = tk.Button(self.model_frame, text="Submit", command=self.submit)
        self.submit_button.pack(side="left")

        self.cluster_frame = tk.Frame(master)
        self.cluster_frame.pack()

        # Add options to choose the cluster method
        self.cluster_label = tk.Label(self.cluster_frame, text="Cluster Method:")
        self.cluster_label.pack(side="left")
        self.cluster_options = ["KMeans", "Hierarchical"]
        self.cluster_var = tk.StringVar(value="KMeans")
        self.cluster_dropdown = tk.OptionMenu(self.cluster_frame, self.cluster_var, *self.cluster_options)
        self.cluster_dropdown.pack(side="left")

        # Add options to choose the linkage method
        self.linkage_label = tk.Label(self.cluster_frame, text="Linkage method:")
        self.linkage_label.pack(side="left")
        self.linkage_options = ["ward", "complete","average", "single"]
        self.linkage_var = tk.StringVar(value="ward")
        self.linkage_dropdown = tk.OptionMenu(self.cluster_frame, self.linkage_var, *self.linkage_options)
        self.linkage_dropdown.pack(side="left")
        
        # Add a checkbox for the clustering type
        self.autoclust_var = tk.BooleanVar(value=True)
        self.autoclust_check = tk.Checkbutton(self.cluster_frame, text="Automatic cluster", variable=self.autoclust_var)
        self.autoclust_check.pack(side="left")

        # Add a checkbox to display the dendrogram
        self.show_dendrogram_var = tk.BooleanVar(value=True)
        self.show_dendrogram_check = tk.Checkbutton(self.cluster_frame, text="Show Dendrogram", variable=self.show_dendrogram_var)
        self.show_dendrogram_check.pack(side="left")

        # Add a checkbox to display the plot for Kmeans cluster options
        self.show_kmeans_plot_var = tk.BooleanVar(value=True)
        self.show_kmeans_plot_check = tk.Checkbutton(self.cluster_frame, text="Show Kmeans plot", variable=self.show_kmeans_plot_var)
        self.show_kmeans_plot_check.pack(side="left")

        # Display the suggested number of clusters
        self.optimal_k_label = tk.Label(self.cluster_frame, text="Optimal Clusters:")
        self.optimal_k_label.pack(side="left")
        self.optimal_k_textbox = tk.Text(self.cluster_frame, height=1, width=5)
        self.optimal_k_textbox.pack(side="left")


        # Create the area to display the DEM plot
        self.fig = Figure(figsize = (5, 5), dpi = 100)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

    def submit(self):
        self.fig.clf()

        # Collect all of the provided variables
        colour_space = self.color_var.get()
        auto_threshold = self.threshold_var.get()
        manual_threshold = None

        if not auto_threshold:
            manual_threshold = int(self.manual_entry.get())

        show_clusters = self.show_clusters_var.get()
        show_cluster_labels = self.show_cluster_labels_var.get()
        show_dendrogram = self.show_dendrogram_var.get()
        show_kmeans_plot = self.show_kmeans_plot_var.get()
        show_contours = self.show_contours_var.get()
        show_surface = self.show_surface_var.get()
        cluster_method = self.cluster_var.get()
        linkage_method = self.linkage_var.get()
        auto_cluster = self.autoclust_var.get()

        file_name = self.file_entry.get()
        
        # Determine the surface model to use
        surface_model = self.surface_var.get()
        if surface_model == "Triangulate Surface":
            trisurf = True
        else:
            trisurf = False
            
        contour_interval = int(self.spinbox.get())
        n_clusters = int(self.n_clusters_entry.get())

        # Find the contours from the image
        image = cv2.imread(file_name)
        contours = extractContours(image, colour_space, manual_threshold)

        # Find the features from the contours
        features = get_features(contours)
        normalised_features = normalise_features(features)

        # Find the clusters from the contours
        if auto_cluster:
            clusters, n, Z = find_clusters(contours, cluster_method, linkage_method=linkage_method)
        else:
            clusters, n, Z = find_clusters(contours, cluster_method, n_clusters=n_clusters, linkage_method=linkage_method)
            n = optimal_kmeans_clusters(normalised_features, show_kmeans_plot)

        # Update the suggested contour number
        self.optimal_k_textbox.delete("1.0", tk.END)
        self.optimal_k_textbox.insert(tk.END, str(n))

        if show_dendrogram and cluster_method == "Hierarchical":
            plt.figure(figsize=(5, 3))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Contour Index')
            plt.ylabel('Distance')
            dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
            plt.show()

        # Display the image of the clusters
        if show_clusters:
            draw_clusters(image, clusters, show_cluster_labels)

        # Create the plot
        ax = self.fig.add_subplot(111, projection='3d')

        # Plot the clusters onto the given axes
        plot_clusters(image, clusters, contour_interval, ax, trisurf, show_contours, show_surface)

        # display plot in canvas
        self.canvas.draw()
    

def main():
    root = tk.Tk()
    my_gui = MyGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
