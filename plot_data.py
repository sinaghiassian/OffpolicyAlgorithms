import os
import numpy

from Plotting.plot_all_sensitivities_per_alg_gradients import plot_all_sensitivities_per_alg_gradients
from Plotting.plot_dist import plot_distribution
from Plotting.plot_all_sensitivities_per_alg_emphatics import plot_all_sensitivities_per_alg_emphatics
from Plotting.plot_learning_curve import plot_learning_curve
from Plotting.plot_learning_curves_for_all_third_params import plot_all_learning_curves_for_one_third
from Plotting.plot_learning_for_two_lambdas import plot_learning_curve_for_lambdas
from Plotting.plot_sensitivity import plot_sensitivity_curve
from Plotting.plot_sensitivity_for_two_lambdas import plot_sensitivity_for_lambdas
from Plotting.plot_specific_learning_curves import plot_specific_learning_curves
from Plotting.plot_waterfall import plot_waterfall_scatter
from Plotting.process_state_value_function import plot_value_functions
from Tasks.HighVarianceLearnEightPoliciesTileCodingFeat import HighVarianceLearnEightPoliciesTileCodingFeat
from Tasks.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
from process_data import process_data

# process_data()
# plot_learning_curve()
# plot_sensitivity_curve()
# plot_waterfall_scatter()
# plot_learning_curve_for_lambdas()
# plot_sensitivity_for_lambdas()
plot_distribution()
# plot_specific_learning_curves()
# plot_all_sensitivities_per_alg_gradients()
# plot_all_sensitivities_per_alg_emphatics()
# plot_value_functions()
# plot_all_learning_curves_for_one_third()


# For building d_mu
# obj = HighVarianceLearnEightPoliciesTileCodingFeat()
# d_mu = (obj.generate_behavior_dist(20_000_000))
# numpy.save(os.path.join(os.getcwd(), 'Resources', 'HighVarianceLearnEightPoliciesTileCodingFeat', 'd_mu.npy'), d_mu)
