#include <iostream>
#include <iomanip>
#include "matrix_lib.hpp"
#include "matrix_lib_tmpl.hpp"

#define FPI           3.14159265358979323846f

// converts from degrees to radians
float dtr(float deg) {return FPI * deg / 180.0f;}

template<typename vT3, typename mT3>
void test()
{
vT3 u(0,-4,2);
vT3 v(-1,2,3);

mT3 S(3,0,0,
      0,1,0,
      0,0,-2);

float rot_deg = 45.0f;
mT3 Rx(1.0f,               0.0f,                0.0f,
       0.0f, cosf(dtr(rot_deg)), -sinf(dtr(rot_deg)),
       0.0f, sinf(dtr(rot_deg)),  sinf(dtr(rot_deg)));
         


std::cout << std::fixed << std::setprecision(2);
std::cout << "u: " << u << "\n";
std::cout << "v: " << v << "\n";
std::cout <<"u+v: " << u+v << "\n";
std::cout <<"u-v: " << u-v << "\n";
std::cout <<"PI*v: " << FPI*v << "\n";
std::cout <<"u dot v: " << dot(u,v) << "\n";
std::cout <<"||u||_1/3: " << norm(u,1.0f/3.0f) << "\n";
std::cout <<"||u||_1  : " << norm1(u) << "\n";
std::cout <<"||u||_2  : " << norm2(u) << "\n";
std::cout <<"||u||_10 : " << norm(u,10.0f) << "\n";
std::cout <<"||u||_inf: " << norminf(u) << "\n";
std::cout <<"u x v: " << cross(u,v) << "\n";
std::cout <<"v x u: " << cross(v,u) << "\n";
std::cout <<"proj_u(v): " << proj(u,v) << "\n";
std::cout <<"proj_v(u): " << proj(v,u) << "\n";


std::cout <<"\n";
std::cout << "Non-uniform scaling matrix, S:\n";
std::cout << S << "\n";
std::cout << "det(S) = " << det(S) << "\n";

std::cout <<"\n";
std::cout << "Rotation matrix around the X axis by " << rot_deg << " degrees, Rx:\n";
std::cout << Rx << "\n";
std::cout << "det(Rx) = " << det(Rx) << "\n";

auto w = Rx * v;
std::cout << "\n";
std::cout << "Rotate v by " << rot_deg << " degrees, w = Rx v:\n" << w << "\n";


std::cout << "\n";
std::cout << "Rotate w by -" << rot_deg << " degrees, w^T Rx ~ Rx^T w = v:\n" << w * Rx << "\n";

std::cout << "\n";
std::cout << "Rx S:\n" << Rx * S << "\n";

std::cout << "\n";
std::cout << "S Rx:\n" << S * Rx << "\n";

std::cout << "\n";
std::cout << "(S Rx)^T:\n" << transpose(S * Rx) << "\n";

std::cout << "\n";
std::cout << "Rx^T S^T:\n" << transpose(Rx) * transpose(S) << "\n";

std::cout << "\n";
std::cout << "(S Rx)^{-1}:\n" << inv(S * Rx) << "\n";

std::cout << "\n";
std::cout << "Rx^{-1}:\n" << inv(Rx) << "\n";

std::cout << "\n";
std::cout << "S^{-1}:\n" << inv(S) << "\n";

std::cout << "\n";
std::cout << "Rx^{-1} S^{-1}:\n" << inv(Rx) * inv(S) << "\n";
}

int main()
{
std::cout << "fmat3_rm\n";
test<fvec3, fmat3_rm>();
std::cout << "\n\n";

std::cout << "fmat3_cm\n";
test<fvec3, fmat3_cm>();
std::cout << "\n\n";

std::cout << "mat_rm\n";
test<vec<float,3>, mat_rm<float,3,3>>();
std::cout << "\n\n";

std::cout << "mat_cm\n";
test<vec<float,3>, mat_cm<float,3,3>>();
std::cout << "\n\n";

return 0;
}