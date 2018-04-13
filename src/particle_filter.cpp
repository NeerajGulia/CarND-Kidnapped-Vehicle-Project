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
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	num_particles = 10;
	for(int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.id = i;
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
		// cout << "init: x: " << p.x << ", y: " << p.y << endl;
	}
	is_initialized = true;
	// cout << "length of particles: " << particles.size() << ", weights length: " << weights.size() << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// cout << "new prediction step" << endl;
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for(auto &p: particles)
	{	
		if (fabs(yaw_rate) == 0.0)
		{
			p.x += delta_t * velocity * cos(p.theta);
			p.y += delta_t * velocity * sin(p.theta);
		}
		else
		{
			double velocityByYaw_rate = (double)velocity/yaw_rate;
			double delta_tIntoYaw_rate = delta_t * yaw_rate;
			double thetaPlusDelta_tIntoYaw_rate = p.theta + delta_tIntoYaw_rate;

			p.x += velocityByYaw_rate * ( sin(thetaPlusDelta_tIntoYaw_rate) - sin(p.theta) );
			p.y += velocityByYaw_rate * ( cos(p.theta) - cos(thetaPlusDelta_tIntoYaw_rate) );
			p.theta += delta_tIntoYaw_rate;
		}
		//Adding uncertainity in the readings
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);

		// cout << "predicted: x: " << p.x << ", y: " << p.y << ", theta: " << p.theta << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	for(auto &obs: observations)
	{
		double minDistance = numeric_limits<double>::max();
		LandmarkObs nearestLandmark;
		int landmarkIndex = 0;
		int index = -1;
		for(auto landMark: predicted)
		{
			index++;
			double distance = dist(landMark.x, landMark.y, obs.x, obs.y);
			if(distance < minDistance)
			{
				landmarkIndex = index;
				minDistance = distance;
			}
		}
		obs.id = landmarkIndex;
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

	int index = 0;
	for(auto &p: particles)
	{
		vector<LandmarkObs> transObservations;
		for(auto obs: observations)
		{
			LandmarkObs ob;
			double cosTheta = cos(p.theta);
			double sinTheta = sin(p.theta);
			ob.x = p.x + cosTheta * obs.x - sinTheta * obs.y;
			ob.y = p.y + sinTheta * obs.x + cosTheta * obs.y;
			transObservations.push_back(ob);
		}

		//get only the landmarks which are in the range of the sensor
		vector<LandmarkObs> validLandmarks;
		for(auto map: map_landmarks.landmark_list)
		{
			double distance = dist(map.x_f, map.y_f, p.x, p.y);
			if(distance <= sensor_range)
			{
				LandmarkObs obj;
				obj.id = map.id_i;
				obj.x = map.x_f;
				obj.y = map.y_f;
				validLandmarks.push_back(obj);
			}
		}

		dataAssociation(validLandmarks, transObservations);
		
		vector<int> associations;
		vector<double> senseX;
		vector<double> senseY;
		double weight = 1.0;
		double firstPart = 1.0/(2*M_PI*std_landmark[0]*std_landmark[1]) ;
		double twoSigmaSquareX = 2*std_landmark[0]*std_landmark[0];
		double twoSigmaSquareY = 2*std_landmark[1]*std_landmark[1];
		for(auto trans_obs: transObservations)
		{
			if(trans_obs.id > -1){
				LandmarkObs validLandmark = validLandmarks[trans_obs.id];
				associations.push_back(validLandmark.id);
				senseX.push_back(validLandmark.x);
				senseY.push_back(validLandmark.y);
				double xPart = trans_obs.x - validLandmark.x;
				double yPart = trans_obs.y - validLandmark.y;
				weight *= firstPart * exp( -( xPart*xPart/twoSigmaSquareX + yPart*yPart/twoSigmaSquareY) );
			}
			else //assign minimum double to particle weight
			{
				weight *= numeric_limits<double>::min();
			}
		}
		// Set associations
		p.associations = associations;
		p.sense_x = senseX;
		p.sense_y = senseY;
		p.weight = weight;
		weights[index] = weight;
		// cout << "weight for index: " << index << ", is: " << weight << endl;
		index++;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> resampledParticles;
	for(int i = 0; i < num_particles; i++)
	{
		int genIndex = d(gen);
		// cout << "i: " << i << ", " << ", P weight: " << particles[i].weight << ", weight : " << weights[i] << 
		// " random index: " << genIndex << ", associations length: " << particles[i].associations.size() << endl;
		resampledParticles.push_back(particles[genIndex]);
	}
	particles = resampledParticles;
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
