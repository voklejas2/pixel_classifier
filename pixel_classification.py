import numpy as np
import scipy
from sklearn.mixture import GaussianMixture

class pixel_classifier:
    """
    Class to classifiy good and bad pixels for a given panel in an image.
    """

    def __init__(self, data, transform_step_size=1, norm_flag=False, max_num_models=10, rescale_data=True,
                 IC_type="AIC", IC_threshold=50, image_shape=[32,128]):
        """
        data(np.matrix): matrix of pixel intensities from 1 detector panel
        transform_step_size(int): step size of bilateral transformation
        norm_flag(bool): flag to perform L2 normalization on pixel intensities before transformation
        max_num_models (int): maximum number of Gaussian distributions in mixture model
        rescale_data (bool): flag to standardize data by standard deviation. (Improves robustnesses of this method)
        IC_type(str): type of information criterion to be used to assess model goodness of fit
        IC_threshold(int): cut-off value used used to assess information crieterion is plateauing
        image_shape (list): size of image shape slow axis dimension is given first, then fast dimension second
        transformed_data (np.array): pixel intensities of single panel after bilateral transform
        X (np.array): transformed pixel intensities reshaped for GaussianMixture object
        gmm_models(obj): Gaussian Mixture model object (sklearn)
        best_model_idx (int): index of gmm_model referring to best performing gmm model
        best_gmm_model (obj): subset of Gaussian Mixture model object (sklearn)   
        means (list): means from best performing Gaussian mixture model
        variances (list): variances from best performing Gaussian mixture model 
        """

        self.data = data
        self.transform_step_size = transform_step_size 
        self.norm_flag = norm_flag
        self.max_num_models = max_num_models
        self.rescale_data = rescale_data
        self.IC_type = IC_type 
        self.IC_threshold = IC_threshold
        self.image_shape = image_shape
        self.transformed_data = self.transform_data()
        if self.rescale_data:
            X = self.transformed_data.flatten()
            X = X/np.std(X)
            self.X = X.reshape(-1,1)
        else:    
            self.X = self.transformed_data.flatten().reshape(-1,1)
        self.gmm_models = self.fit_gaussian_mixture_models()
        self.best_model_idx = self.determine_best_mixture_model()
        self.best_gmm_model = self.gmm_models[self.best_model_idx]
        self.means, self.variances = self.get_gmm_parameters() 

    def transform_data(self):
        """
        Performs bilaterial transformation on panel image data
        """

        if self.norm_flag:
            data = self.data/scipy.linalg.norm(self.data, 2)
            shifted_img = scipy.ndimage.shift(data,
                                     self.transform_step_size,
                                     order=0,
                                     mode='nearest'
                                     )
            trans_img = data - shifted_img

        shifted_img = scipy.ndimage.shift(self.data,
                                     self.transform_step_size,
                                     order=0,
                                     mode='nearest'
                                     )
        trans_img = self.data - shifted_img
        
        return trans_img

    def fit_gaussian_mixture_models(self):
        """
        Performs expectation-maximization algorithm for fitting mixture-of-Gaussian models.
        A Gaussian mixture model is a probabilistic model that assumes all the data points
        are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
        Args:
            X (np.array): 1D array of transformed pixel intensities
            max_num_models (int): maximum number of Gaussian distributions in mixture model
            rescale_data (bool): Flag to standardize data by standard deviation. (Improves robustnesses of this method)
        Returns:
            models (obj.); Gaussian Mixture model object (sklearn)
        """
        num_gaussians_list = np.arange(1,self.max_num_models+1)

        models = [None for i in range(len(num_gaussians_list))]

        for i in range(len(num_gaussians_list)):
            models[i] = GaussianMixture(num_gaussians_list[i]).fit(self.X)
        print('model len',len(models))
        return models

    def determine_best_mixture_model(self):
        """
        Uses Aikake or Bayesian information criterion to determing which Gaussian
        mixture model best describes transformed pixel intensities
        """
        if self.IC_type == "AIC":
            IC = [m.aic(self.X) for m in self.gmm_models]
        elif self.IC_type == "BIC":
            IC = [m.bic(self.X) for m in self.gmm_models]
        else:
            print(f"Only 2 types of information criterion are allowed; \"AIC\" or \"BIC\"")
            print(f"You entered {self.IC_type}")
        diff = [np.abs(IC[n]-IC[n-1]) for n in range(1,len(IC))]
        thresholded_diff = np.where(np.asarray(diff)<self.IC_threshold)

        return int(thresholded_diff[0][0]-1)

    def get_gmm_parameters(self):
        """
        Convenience function that returns 2 lists containing
        the means and variances for distributions determined by
        the best Gaussian mixture model
        """
        means = []
        var = []
        for i in range(self.best_model_idx+1):
            means.append(self.best_gmm_model.means_[i][0])
            var.append(self.best_gmm_model.covariances_[i][0][0])
        print('EM algorithm converged?:', self.best_gmm_model.converged_)
        print(f'Found {len(means)} different distributions in best Gaussian mixture model')
        print(f'means: {means}')
        print(f'variances: {var}')
        return means, var

    def label_bad_pixels(self):
        """
        Predicts which Gaussian distribution each pixel belongs. Specifically,
        the max value of the estimated weighted log probability for each
        transformed pixel intensity in X is calculated.
        Note: Transformed pixels with values of exactly zero are assumed to be dead
              or otherwise in a permanently "bad" state
        Returns:
            classed_img(np.matrix): matrix of the same shape as input panel image where each
                                    element corresponds to a pixel that is labeled good (0) or
                                    bad (10). 
        """
        mmin = np.min(np.abs(self.means))
        if mmin in self.means:
            center_mean = mmin
        elif -1*mmin in self.means:
            center_mean = -1*mmin

        good_distribution_idx = int(np.where(np.asarray(self.means)==center_mean)[0])
        classed = self.best_gmm_model.fit_predict(self.X)
        bad_pixel_idx = np.where(classed!=good_distribution_idx)[0]
        dead_pixels = np.where(self.X==0.0)[0]
        for dp in dead_pixels:
            classed[dp] = 10
        for bp in bad_pixel_idx:
            classed[bp] = 10
        classed_img = classed.reshape(self.image_shape[0], self.image_shape[1])

        return classed_img
