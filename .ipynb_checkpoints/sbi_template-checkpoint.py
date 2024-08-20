import torch
from typing import Dict, List, Optional, Callable
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn


def flatten_observations(obs: Dict[str, torch.tensor]) -> torch.tensor:
    """Flatten multiple observations into a single data vector"""
    return torch.concat([torch.flatten(v) for k,v in obs.items])    

def calibrate_inits(inits: torch.tensor, params: Optional[torch.tensor] = None) -> torch.tensor:
    """Apply calibrations to initial condition"""
    if params is not None:
        # TODO: Apply calibrations with given parameters
        raise NotImplementedError
    return inits

def calibrate_forcings(forcings: torch.tensor, params: Optional[torch.tensor] = None) -> torch.tensor:
    """Apply calibrations to forcings"""
    if params is not None:
        # TODO: Apply calibrations with given parameters
        raise NotImplementedError
    return forcings

def calibrate_observations(
    observations: Dict[str, torch.tensor], 
    params: Optional[torch.tensor] = None
) -> Dict[str, torch.tensor]:
    """Apply calibrations to observations"""
    if params is not None:
        # TODO: Apply calibrations with given parameters
        raise NotImplementedError
    return observations

def split_parameters(params: torch.tensor) -> List[torch.tensor]:
    """Split full parameter vector into parameters for simulator, forcing calibration, initial condition calibration, and prediction calibration
    """
    # TODO: actual split if needed
    return params[0:], None, None, None

def run_simulation(simulator: ParflowHook, theta: torch.tensor, inits: torch.tensor, forcings: torch.tensor, observables: List[str]) -> Dict[str, torch.tensor]:
    """Run ParFlow simulation with given parameters, initial conditions and forcings.

    Parameters:
    -----------
    simulator:
        hook to call parflow for a preset domain and timeframe
    theta:
        vector of all relevant model parameters
    inits:
        initial conditions for 3D pressure field
    forcings:
        forcing data as need by parflow.
        TODO: does forcing have information about time or does model have that information???
    observables:
        list of strings that explain which observations to create (type and location)
    
    Returns:
    --------
    dict of predictions, with the same keys as observables
    """
    sim_params, forcing_params, init_params, pred_params = split_parameters(theta)
    inits = calibrate_inits(inits, init_params)
    simulator.set_initial_conditions(inits)
    simulator.set_parameters(sim_params)
    forcings = calibrate_forcings(forcings, forcing_params)
    pressures = simulator.run(forcings)
    pred = simulator.compute_predictions(pressures, observables)
    pred = calibrate_predictions(pred, pred_params)
    return pred
    
def make_prior(parameters: List[str]) -> sbi.Distribution:
    """Make prior distribution for all parameters

    Parameters:
    -----------
    parameters:
        list of parameter names to construct prior for

    Returns:
    --------
    sbi.Distribution
    """
    raise NotImplementedError

def anpe(
    simulator: ParflowHook, 
    huc: int, 
    times: torch.tensor, 
    parameters: List[str], 
    observables: List[str], 
    num_sims: int = 1000,
) -> sbi.PosteriorEstimator:
    """Compute amortized posterior for parameters given observables.

    This estimator can be reused for new observations _as long as no other aspects (such as forcings) are changed_.

    Parameters:
    -----------
    simulator:
        hook to call parflow for a preset domain and timeframe
    huc:
        number to indicate domain
    times:
        list of times for model forcings and pressure calculation
        TODO: are these specified as a list or some other structure???
    parameters:
        list of parameter names to construct prior for
    observables:
        list of strings that explain which observations to create (type and location)
    num_sims:
        number of simulations to train neural density estimator
    
    Returns:
    --------
    neural posterior estimator
    """
    # HydroData and Parflow interfaces
    forcings = get_forcings(huc, times)
    inits = get_initial_conditions(huc, times[0])
    simulator.setup(huc, times)

    # SBI setup
    prior = make_prior(parameters)
    inference = SNPE(prior)
    # create simulator function call that only depends on parameters, everything else is fixed
    train_simulator = lambda theta: flatten_observations(run_simulation(simulator, theta, inits, forcings, observables))
    simulator, prior = prepare_for_sbi(train_simulator, prior)
    
    # run simulations and store flattened predictions
    # thetas: (nsim, ndim)
    # xs: (nsim, npred = length of flattened predicted data vector)
    thetas, xs = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sims)
    
    # TODO: potentially compress predicted timeseries
    # xs = compress(xs) 

    # train neural density estimator for posterior
    density_estimator = inference.append_simulations(thetas, xs).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior

def snpe(
    simulator: ParflowHook, 
    huc: int, 
    times: torch.tensor, 
    parameters: List[str], 
    observables: List[str],
    observations: Dict[str, torch.tensor],
    num_rounds: int = 10,
    num_sims: int = 1000,
    num_samples: int =10_000
) -> sbi.PosteriorEstimator:
    """Compute sequential posterior estimator for parameters given a specific set of observations.

    This estimator is faster to train than anpe() but only applicable to a specific set of observations.

    Parameters:
    -----------
    simulator:
        hook to call parflow for a preset domain and timeframe
    huc:
        number to indicate domain
    times:
        list of times for model forcings and pressure calculation
        TODO: are these specified as a list or some other structure???
    parameters:
        list of parameter names to construct prior for
    observables:
        list of strings that explain which observations to create (type and location)
    observations:
        dict of observations, with the same keys as observables
    num_rounds:
        number of sequential updates to prior
    num_sims:
        number of simulations per round
    num_samples:
        number of samples to refine estimator
    
    Returns:
    --------
    neural posterior estimator
    """
    # HydroData and Parflow interfaces
    forcings = get_forcings(huc, times)
    inits = get_initial_conditions(huc, times[0])
    simulator.setup(huc, times)

    # SBI setup
    prior = make_prior(parameters)
    inference = SNPE(prior)
    # create simulator function call that only depends on parameters, everything else is fixed
    train_simulator = lambda theta: flatten_observations(run_simulation(simulator, theta, inits, forcings, observables))
    x_o = flatten_observations(observations)

    # run simulations and iteratively refine parameter distribution
    proposal = prior
    for _ in range(num_rounds):
        thetas = proposal.sample((num_sims,))
        xs = train_simulator(thetas)
        _ = inference.append_simulations(thetas, xs).train(force_first_round_loss=True)
        # check results agains observations and reject parameters that don't predict anything close to them
        posterior = inference.build_posterior().set_default_x(x_o)
        accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4, num_samples_to_estimate_support=num_samples)
        # update prior for next round
        proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
    return posterior

# simulator: callable function, Parflow or Emulator
# huc: identified
# times: timestamp range
# parameters: list of labels to explore in SBI, e.g. Mannings in cell-type 1
# observables: list of labels to indentify the observation, e.g. streamflow of gauge X

# amortized posterior
posterior = anpe(simulator, huc, times, parameters, observables)
# observations: dictionary (label -> tensor) to map observable lables to observed values
observations = get_observations(huc, times, observables)
x_o = flatten_observations(observations)
# evaluate estimator at given x_o
samples = posterior.sample((10000,), x=x_o).detach()

# sequential posterior: can work with crude prior and fewer sims
observations = get_observations(huc, times, observables)
posterior = snpe(simulator, huc, times, parameters, observables, observations)
samples = posterior.sample((10000,)).detach()
