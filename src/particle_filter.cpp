/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//default_random_engine gen;
	
	// gausian distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 100;
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	is_initialized = true;
	cout << "Initialized" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	if (abs(yaw_rate) > 0.000001) 
	{
		for (auto&& p : particles)
		{
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta)) + noise_x(gen);
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t)) + noise_y(gen);
			p.theta += yaw_rate * delta_t + noise_theta(gen);
		}
	}
	else
	{
		for (auto&& p : particles)
		{
			p.x += velocity * delta_t * cos(p.theta) + noise_x(gen);
			p.y += velocity * delta_t * sin(p.theta) + noise_y(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto&& obs : observations)
	{
		obs.id = -1;
		double min = numeric_limits<double>::max();
		for (auto pred : predicted)
		{
			double diff = dist(pred.x, pred.y, obs.x, obs.y);
			if (diff < min)
			{
				min = diff;
				obs.id = pred.id;
			}
		}
		//cout << "obs.id " << obs.id << " ";
		// cout << "obs " << obs.x << " " << obs.y << " <-> " << obs.id << "\t";
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// for each particle...
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));
	for (int i = 0; i < num_particles; i++) {

		// get the particle x, y coordinates
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;
		for (auto lmk : map_landmarks.landmark_list)
		{
			// consider landmark within sensor range
			if (sqrt(pow((double)lmk.x_f - p_x, 2) + pow((double)lmk.y_f - p_y, 2)) < sensor_range)
			{
				predictions.push_back(LandmarkObs{ lmk.id_i, (double)lmk.x_f, (double)lmk.y_f });
			}
		}

		// transform observations to map coordinates
		vector<LandmarkObs> map_observations;
		for (auto& obs : observations)
		{
			double x_m = p_x + (cos(p_theta) * obs.x) - (sin(p_theta) * obs.y);
			double y_m = p_y + (sin(p_theta) * obs.x) + (cos(p_theta) * obs.y);
			map_observations.push_back(LandmarkObs{ obs.id, x_m, y_m });
		}

		// perform dataAssociation for the predictions and transformed observations on current particle
		dataAssociation(predictions, map_observations);

		// reinit weight
		particles[i].weight = 1.0;
		// calculate and update weight for each transformed observation
		for (auto map_obs : map_observations)
		{
			double pr_x, pr_y;
			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == map_obs.id) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}
			//cout << "map_obs.id " << map_obs.id << " predictions[map_obs.id].id " << predictions[map_obs.id].id << "\t";
			double exponent = pow((map_obs.x - pr_x), 2) / (2 * pow(sig_x, 2)) + pow((map_obs.y - pr_y), 2) / (2 * pow(sig_y, 2));
			//cout << map_obs.id << " pred " << predictions[map_obs.id].x << " " << predictions[map_obs.id].y << "obs " << map_obs.x << " " << map_obs.y << "\t";
			//cout << "exp " << exponent << " ";
			// calculate weight using normalization terms and exponent
			particles[i].weight *= gauss_norm * exp(-exponent);
			//cout << "obs_weight " << gauss_norm * exp(-exponent) << "\n";
		}
		weights[i] = particles[i].weight;
		//cout
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//    
	std::default_random_engine generator;
	std::discrete_distribution<int> ind(this->weights.begin(), this->weights.end());

	vector<Particle> resampled;

	for (int i = 0; i < num_particles; i++) {

		const int idx = ind(generator);

		Particle p;
		p.id = idx;
		p.x = particles[idx].x;
		p.y = particles[idx].y;
		p.theta = particles[idx].theta;
		p.weight = particles[idx].weight;

		resampled.push_back(p);
	}

	particles.swap(resampled);

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
