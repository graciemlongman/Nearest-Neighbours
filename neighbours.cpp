#include <algorithm>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <numeric>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

// already given function to read in a list of 3D coordinates from an .xyz file
// input: the name of the file
std::vector < std::vector < double > > read_xyz_file(std::string filename, int& N, double& L){

  // open the file
  std::ifstream xyz_file(filename);

  // read in the number of atoms
  xyz_file >> N;
  
  // read in the cell dimension
  xyz_file >> L;
  
  // now read in the positions, ignoring the atomic species
  std::vector < std::vector < double > > positions;
  std::vector < double> pos = {0, 0, 0};
  std::string dummy; 
  for (int i=0;i<N;i++){
    xyz_file >> dummy >> pos[0] >> pos[1] >> pos[2];
    positions.push_back(pos);           
  }
  
  // close the file
  xyz_file.close();
  
  return positions;
  
}

////////////////////////////////////////////////////////////////////////
//                      count neighbours SERIAL                       //
////////////////////////////////////////////////////////////////////////

std::vector < int > get_num_neighbours_serial(std::vector < std::vector < double > > positions, double N, double rc){
  
  double exponent = 2;
  std::vector < int > neighbours(N, 0);

  //loop thorugh the particles
  for (int i=0; i<N; i++){
    for (int j=i+1; j<N; j++){
      // calc each distance between particle in x, y and z direction
      double x = (positions[i][0] - positions[j][0]);
      double y = (positions[i][1] - positions[j][1]);
      double z = (positions[i][2] - positions[j][2]);
      
      double dist_sq = pow(x, exponent) + pow(y, exponent) + pow(z, exponent); 

      if (dist_sq < pow(rc, exponent)){
        neighbours[i] += 1;
        neighbours[j] += 1;
      }
    }
  }

  return neighbours;

}


////////////////////////////////////////////////////////////////////////
//              count neighbours MPI - load distribution             //
////////////////////////////////////////////////////////////////////////

std::vector < int > get_num_neighbours_mpi1(std::vector < std::vector < double > > positions, double N, double rc, int nproc, int iproc){
  
  std::vector < int > neighbours(N, 0);

  //distribute the processors. 
  int di = floor(N / nproc); //how much target load each task gets
  int total = 0;

  // creat a vector filled with values of i to divide the atoms up between tasks
  std::vector <int> load;
  for (int i = N-1; i>0; --i){
    total += i;
    if(total>di){
      load.push_back(i);
      total = 0;
    }
  }
  
  //calculate loop start point
  int i0;
  if (iproc == nproc -1){
    i0 = 0;
  } 
  else {
   i0 = N - load[iproc-1];
  }
  
  //calculate loop end point
  int i1;
  if (iproc == 0){
    i1 = (N-1);
  } 
  else { 
    i1 = N - load[iproc] -1;
  }


  double exponent = 2;
  //loop thorugh the particles i and the other particles, j = i+1
  for (int i=i0; i<i1; i+=1){
    for (int j=i+1; j<N; j+=1){
        // calc each distance between particle in x, y and z direction
        double x = (positions[i][0] - positions[j][0]);
        double y = (positions[i][1] - positions[j][1]);
        double z = (positions[i][2] - positions[j][2]);
        
        double dist_sq = pow(x, exponent) + pow(y, exponent) + pow(z, exponent); 

        if (dist_sq < pow(rc, exponent)){
          neighbours[i] += 1;
          neighbours[j] += 1;
      }
    }
  }
  
  //init vector to store the reduced neighbours into
  std::vector<int> global_neighbours(N,0);

  MPI_Reduce(neighbours.data(), global_neighbours.data(), N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  return global_neighbours;

}


////////////////////////////////////////////////////////////////////////
//                        MPI - 2                                     //
////////////////////////////////////////////////////////////////////////

std::vector < int > get_num_neighbours_mpi2(std::vector < std::vector < double > > positions, double N, double rc, int nproc, int iproc){

  double exponent = 2;
  std::vector < int > neighbours(N, 0);

  //distribute the processors. 
  int n = N / nproc;
  int remainder = n % nproc;
  
  int start = iproc * n + std::min(iproc,remainder);
  int end = start + n;

  if (iproc < remainder){
    end+=1;
  }
 
  //loop thorugh the particles
  for (int i=start; i<end; i++){
    for (int j=i+1; j<N; j++){
        // calc each distance between particle in x, y and z direction
        double x = (positions[i][0] - positions[j][0]);
        double y = (positions[i][1] - positions[j][1]);
        double z = (positions[i][2] - positions[j][2]);
        
        double dist_sq = pow(x, exponent) + pow(y, exponent) + pow(z, exponent); 

        if (dist_sq < pow(rc, exponent)){
          neighbours[i]++;
          neighbours[j]++;
      }
    }
  }
  
  //init vector to store the reduced neighbours into
  std::vector<int> global_neighbours(N,0);

  MPI_Reduce(neighbours.data(), global_neighbours.data(), N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  return global_neighbours;

}


////////////////////////////////////////////////////////////////////////
//                              main                                  //
////////////////////////////////////////////////////////////////////////

int main(int argc, char**argv){

  // initialise MPI
  MPI_Init(&argc, &argv);

  int nproc, iproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc); //get num processes
  MPI_Comm_rank(MPI_COMM_WORLD, &iproc); //get rank of process


  // read in filename, num atoms, N, box length, L
  std::string filename = argv[1];
  int N = atoi(argv[2]);
  double L = atof(argv[3]);
  double rc = 9.0;

  //read in xyz file
  std::vector < std::vector < double > > positions = read_xyz_file(filename, N, L);
  
  // vectors not guaranteed to be the same across tasks. so bcast from root task
  MPI_Barrier(MPI_COMM_WORLD);
  
  //calc num of nearest nieghbours for each particle
  std::vector < int > neighbours1, neighbours2, neighbours3;

  //start timer - serial approach
  double start1 = MPI_Wtime();
  neighbours1 = get_num_neighbours_serial(positions, N, rc);
  //end timer
  double end1 = MPI_Wtime();

  //start timer - block approach
  double start2 = MPI_Wtime();
  neighbours2 = get_num_neighbours_mpi1(positions, N, rc, nproc, iproc);
  //end timer
  double end2 = MPI_Wtime();

  //start timer - cyclic approach
  double start3 = MPI_Wtime();
  neighbours3 = get_num_neighbours_mpi2(positions, N, rc, nproc, iproc);
  //end timer
  double end3 = MPI_Wtime();

  // calc min, max and average
  double sum = std::accumulate(neighbours1.begin(), neighbours1.end(), 0.0);
  double avg = sum / neighbours1.size();
  double min = *std::min_element(neighbours1.begin(), neighbours1.end());
  double max = *std::max_element(neighbours1.begin(), neighbours1.end());
 
   for (int i: neighbours1)
     std::cout << i << "," << std::endl; 

  //cmd line output
  std::cout << "****" << std::endl;
  std::cout << "time" << "," << nproc << "," << N << "," << end1 - start1 << "," << end2 - start2 << "," << end3 - start3 << std::endl; 
  std::cout << "stats" << avg << "," << min << "," << max << "," << std::endl;

  //Finalize MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}
