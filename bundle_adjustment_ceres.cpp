#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
    //     return 1;
    // }
// 取消了命令行传入 文件名参数（"../problem-16-22106-pre.txt"）
// 测试方法：在build2文件夹中1、cmake..     2、make -j8     3、./bundle_adjustment_ceres

    // BALProblem bal_problem(argv[1]);
    //   创建 BALProblem bal_problem 的类；
    //   传入的参数用于类的构造函数：：：将原始数据写入bal_problem中的各个参数！！
    BALProblem bal_problem("../problem-16-22106-pre.txt", false);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);

    bal_problem.WriteToPLYFile("initial.ply");  // 将原始数据写入ply文件

    SolveBA(bal_problem);      // ！！将原始数据传入SolveBA，进行BA！！
    
    bal_problem.WriteToPLYFile("final.ply");    // 将BA后的数据写入ply文件

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();    // 3
    const int camera_block_size = bal_problem.camera_block_size();  // use_quaternions_ ? 10 : 9;
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i] respectively.
        // 每个观测对应于一对相机和一个点，它们分别由camera_index（）[i]和point_index）[i]标识。
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        // 每轮循环 都向 ceres::Problem的ResidualBlock 添加 相机和点的参数
        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
}