//travel_time_marcher_genes.cpp

#include "travel_time_marcher_genes.h"
#include "math.h"
#include "heap.h"
#include <stdexcept>
#include <vector>
#include <algorithm>    // std::min_element, std::max_element
#include <cmath>
#include <map>
using std::vector;

#include <iostream>


void travelTimeMarcherGenes::initalizeFrozen()
{
  distanceMarcher::initalizeFrozen();
  for (int i=0; i<size_; i++)
  {
    if (flag_[i]==Frozen)
    {
      // convert distance to time
      distance_[i] = fabs(distance_[i]/speeds_[branch_[i] * size_ + i]);
      // note: branch_[i] should be zero-initialised for frozen nodes!
    }
  }
}

double travelTimeMarcherGenes::updatePointOrderTwo(int i)
{
    double res = updatePointOrderTwo(i, std::set<int>());
    if (res == std::numeric_limits<double>::infinity()) {
        throw std::runtime_error("Unreachable voxel");
    } else {
        return res;
    }
}

// second order point update
// update the distance from the frozen points
const double aa         =  9.0/4.0;
const double oneThird   =  1.0/3.0;
double travelTimeMarcherGenes::updatePointOrderTwo(int i, std::set<int>avoid_dim)
{
  // try to solve for all possible branch values when neighbours
  // have different values, and choose the one that has the smallest solution
  // for distance. Rationale: we want the first arrival to win, and to solve for
  // the smallest arrival time. Varying the branch value to look at all our
  // neighbours is like varying the path near the end-point (when the end-point
  // is near a caustic/boundary).

  // first get neighbouring branch values and iterate over sets of neighbours
  // from the same subclone

  vector<unsigned int> branch_values = get_neighbouring_branch_values(i);
  map<unsigned int, double> tau_values;

  for (auto& branch : branch_values) {
    double a,b,c;
    a=b=c=0;
    int naddr, naddr2; // addresses of neighbours
    // Choose a "good" pair of neighbours on different axes:
    for (int dim=0; dim<dim_; dim++) {
      if (avoid_dim.find(dim) != avoid_dim.end()) {
        continue; //we should avoid this dimension
      }
      double value1 = maxDouble;
      double value2 = maxDouble;
      for (int j = -1; j < 2; j += 2) // each direction (e.g. left and right)
      {
        naddr = _getN(i, dim, j, Mask); // get the neighbour of i along dim
        if (branch_[naddr] != branch) {
          continue; // only consider neighbours with branch_[nbr] == branch
        }
        if (naddr!=-1 && flag_[naddr]==Frozen)
        {
          if (fabs(distance_[naddr])<fabs(value1))
          {
            value1 = distance_[naddr];
            naddr2 = _getN(i, dim, j * 2, Mask);
            if (naddr2 != -1 &&
                flag_[naddr2]==Frozen &&
                ((distance_[naddr2]<=value1 && value1 >=0) ||
                 (distance_[naddr2]>=value1 && value1 <=0)))
            {
              value2=distance_[naddr2];
              if (phi_[naddr2] * phi_[naddr] < 0  || phi_[naddr2] * phi_[i] < 0)
                value2 *= -1;
            }
          }
        }
      }
      if (value2<maxDouble)
      {
        double tp = oneThird*(4*value1-value2);
        a+=idx2_[dim]*aa;
        b-=idx2_[dim]*2*aa*tp;
        c+=idx2_[dim]*aa*pow(tp,2);
      }
      else if (value1<maxDouble)
      {
        a+=idx2_[dim];
        b-=idx2_[dim]*2*value1;
        c+=idx2_[dim]*pow(value1,2);
      }
    }

    // TODO instead of choosing a branch_ value out here, get all the neighbours'
    // branch values, and see which one results in the soonest/shortest
    // time/distance.

    // solve quadratic and find the tau and b values that minimise tau:
    branch_[i] = branch; // set initial branch value
    try {
      double res = solveQuadratic(i,a,b,c);
      // update branch function if a driver mutation is present at site i
      // AND the mutation is not already accounted for
      branch_[i] |= drivers_[i];
      tau_values[branch] = res;
    } catch (std::runtime_error & err) {
      //if the determinant is negative, we try to reach the voxel with one dimension less and take the minimum
      if (avoid_dim.size() == (size_t)dim_) {
        //end of the recursion, use inf so that it is discarded selecting the minimum
        tau_values[branch] = std::numeric_limits<double>::infinity(); 
      }
      vector<double> sols;
      for (int ind=0; ind<dim_; ind++){
        //remove one dimension more than what we are already doing
        std::set<int> tempset = avoid_dim;
        std::pair<std::set<int>::iterator, bool> ret = tempset.insert(ind);
        //avoid recursive call on identical parameters (the set already had *ind* in it):
        if(!ret.second) continue;
        sols.push_back(updatePointOrderTwo(i,tempset));
      }
      if (sols.size()==0) {
        tau_values[branch] = std::numeric_limits<double>::infinity();
        //All the derivates with different dimensionalities are 0
      }
      // update branch function if a driver mutation is present at site i
      // AND the mutation is not already accounted for
      branch_[i] |= drivers_[i];
      tau_values[branch] = *std::min_element(sols.begin(), sols.end());
    }
  }

  // DEBUG:
  if (branch_values.size() > 1) {
    for (auto& branch : branch_values) {
      std::cout << branch << " ";
      std::cout << tau_values[branch] << std::endl;
    }
  }

  // look through the map of b and tau values and return the minimal tau 
  double best_tau = maxDouble;
  for (const auto& [branch, tau] : tau_values) {
    if (tau < best_tau) {
      branch_[i] = branch;
      best_tau = tau;
    }
  }
  // update branch function if a driver mutation is present at site i
  // AND the mutation is not already accounted for
  branch_[i] |= drivers_[i];

  return best_tau;
}

vector<unsigned int> travelTimeMarcherGenes::get_neighbouring_branch_values(int site) {
  vector<unsigned int> neighbouring_branch_values;
  for (int dim=0; dim<dim_; dim++) {
    for (int j=-1; j<2; j+=2) // each direction (e.g. left and right)
    {
      int naddr = _getN(site, dim, j, Mask); // get the neighbour of i along dim
      if ((naddr!=-1) && (flag_[naddr]==Frozen) && std::isfinite(distance_[naddr])) {
        neighbouring_branch_values.push_back(branch_[naddr]);
      }
    }
  }
  return neighbouring_branch_values;
}

double travelTimeMarcherGenes::solveQuadratic(int i, const double &a,
                                         const double &b,
                                         double &c)
{
  double r0 = maxDouble;
  double c2 = c;
  c2 -= 1/pow(speeds_[branch_[i] * size_ + i], 2);
  double det = pow(b, 2) - 4 * a * c2;
  if (det >= 0)
  {
    if ((-b + sqrt(det)) / 2.0 / a < r0){
      r0 = (-b + sqrt(det)) / 2.0 / a;
    }
  }
  
  if (r0 >= maxDouble) {
    throw std::runtime_error("Negative discriminant in (genetic) time marcher quadratic.");
  }
  return r0;
}
