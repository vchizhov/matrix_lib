#pragma once
#include <cmath>
#include <ostream>

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//                         3D Single Precision Vector                        //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

struct fvec3
{
    float e[3];
    
    fvec3() : e{0,0,0} {}
    fvec3(float e0, float e1, float e2) : e{e0,e1,e2} {}

    const float& operator[](int i) const {return e[i];}
    float& operator[](int i) {return e[i];}
};

//---------------------------------------------------------------------------//

fvec3 operator+(const fvec3& lhs, const fvec3& rhs)
{
    return fvec3(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

//---------------------------------------------------------------------------//

fvec3 operator-(const fvec3& lhs, const fvec3& rhs)
{
    return fvec3(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
}

//---------------------------------------------------------------------------//

fvec3 operator*(float lhs, const fvec3& rhs)
{
    return fvec3(lhs*rhs[0], lhs*rhs[1], lhs*rhs[2]);
}

//---------------------------------------------------------------------------//

fvec3 operator*(const fvec3& lhs, float rhs)
{
    return fvec3(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
}

//---------------------------------------------------------------------------//

fvec3 operator/(const fvec3& lhs, float rhs)
{
    return fvec3(lhs[0]/rhs, lhs[1]/rhs, lhs[2]/rhs);
}

//---------------------------------------------------------------------------//

float dot(const fvec3& lhs, const fvec3& rhs)
{
    return lhs[0]*rhs[0]+lhs[1]*rhs[1]+lhs[2]*rhs[2];
}

//---------------------------------------------------------------------------//

float norm1(const fvec3& arg)
{
    return abs(arg[0]) + abs(arg[1]) + abs(arg[0]);
}

//---------------------------------------------------------------------------//

float norm2(const fvec3& arg)
{
    return sqrtf(dot(arg,arg));
}

//---------------------------------------------------------------------------//

float norminf(const fvec3& arg)
{
    return std::max(std::max(arg[0],arg[1]),arg[2]);
}

//---------------------------------------------------------------------------//

float norm(const fvec3& arg, float p)
{
    return powf(powf(abs(arg[0]),p) + powf(abs(arg[1]),p) + powf(abs(arg[2]),p),1.0f/p);
}

//---------------------------------------------------------------------------//

fvec3 cross(const fvec3& lhs, const fvec3& rhs)
{
    return fvec3(lhs[1]*rhs[2]-lhs[2]*rhs[1],
                 lhs[2]*rhs[0]-lhs[0]*rhs[2],
                 lhs[0]*rhs[1]-lhs[1]*rhs[0]);
}

//---------------------------------------------------------------------------//

fvec3 proj(const fvec3& onto, const fvec3& arg)
{
    return dot(onto, arg)/dot(onto,onto) * onto;
}

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& lhs, const fvec3& rhs)
{
    lhs << "[" << rhs[0] << "," << rhs[1] << "," << rhs[2] << "]";
    return lhs;
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//                   3x3 Single Precision Row-Major Matrix                   //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

struct fmat3_rm
{
    fvec3 e[3];
    
    fmat3_rm() {}
    fmat3_rm(float e00, float e01, float e02, 
          float e10, float e11, float e12, 
          float e20, float e21, float e22) 
         : e{fvec3(e00,e01,e02), fvec3(e10,e11,e12), fvec3(e20,e21,e22)} {}
    fmat3_rm(const fvec3& e0, const fvec3& e1, const fvec3& e2) : e{e0,e1,e2} {}

    // return the i-th row as a vector
    const fvec3& operator[](int i) const {return e[i];}
    fvec3& operator[](int i) {return e[i];}

    // return the (i,j)-th element of a matrix
    const float& operator()(int i, int j) const {return e[i][j];}
    float& operator()(int i, int j) {return e[i][j];}
};

//---------------------------------------------------------------------------//

fmat3_rm operator+(const fmat3_rm& lhs, const fmat3_rm& rhs)
{
    return fmat3_rm(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_rm operator-(const fmat3_rm& lhs, const fmat3_rm& rhs)
{
    return fmat3_rm(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_rm operator*(float lhs, const fmat3_rm& rhs)
{
    return fmat3_rm(lhs*rhs[0], lhs*rhs[1], lhs*rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_rm operator*(const fmat3_rm& lhs, float rhs)
{
    return fmat3_rm(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
}

//---------------------------------------------------------------------------//

fmat3_rm operator/(const fmat3_rm& lhs, float rhs)
{
    return fmat3_rm(lhs[0]/rhs, lhs[1]/rhs, lhs[2]/rhs);
}

//---------------------------------------------------------------------------//

fvec3 operator*(const fmat3_rm& lhs, const fvec3& rhs)
{
	//       [ M(1,:) dot v ]
	// M v = [ M(2,:) dot v ]
	//       [ M(3,:) dot v ]
    return fvec3(dot(lhs[0],rhs), dot(lhs[1],rhs), dot(lhs[2],rhs));
}

//---------------------------------------------------------------------------//

fvec3 operator*(const fvec3& lhs, const fmat3_rm& rhs)
{
	// v^T M = v(1) M(1,:) + v(2) M(2,:) + v(3) M(3,:)
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}

//---------------------------------------------------------------------------//

fmat3_rm operator*(const fmat3_rm& lhs, const fmat3_rm& rhs)
{
	//       [ A(1,:) B ]
	// A B = [ A(2,:) B ]
	//       [ A(3,:) B ]
    return fmat3_rm(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
}

//---------------------------------------------------------------------------//

float det(const fmat3_rm& arg)
{
	// Laplace expansion
    return dot(arg[0],cross(arg[1],arg[2]));
}

//---------------------------------------------------------------------------//

fmat3_rm transpose(const fmat3_rm& arg)
{
    return fmat3_rm(arg(0,0), arg(1,0), arg(2,0),
                 arg(0,1), arg(1,1), arg(2,1),
                 arg(0,2), arg(1,2), arg(2,2));                 
}

//---------------------------------------------------------------------------//

fmat3_rm inv(const fmat3_rm& arg)
{
    fmat3_rm res;
	// cofactor matrix
    res[0] = cross(arg[1],arg[2]);
    res[1] = cross(arg[2],arg[0]);
    res[2] = cross(arg[0],arg[1]);
    float det = dot(arg[0],res[0]);
	// M^{-1} = cof(M)^T / det(M) = adj(M) / det(M)
    return transpose(res)/det;
}

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& lhs, const fmat3_rm& rhs)
{
    lhs << "[" << rhs[0] << ",\n " << rhs[1] << ",\n " << rhs[2] << "]";
    return lhs;
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//                   3x3 Single Precision Column-Major Matrix                //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

struct fmat3_cm
{
    fvec3 e[3];
    
    fmat3_cm() {}
    fmat3_cm(float e00, float e01, float e02, 
             float e10, float e11, float e12, 
             float e20, float e21, float e22) 
         : e{fvec3(e00,e10,e20), fvec3(e01,e11,e21), fvec3(e02,e12,e22)} {}
    fmat3_cm(const fvec3& e0, const fvec3& e1, const fvec3& e2) : e{e0,e1,e2} {}

    // return the i-th column as a vector
    const fvec3& operator[](int i) const {return e[i];}
    fvec3& operator[](int i) {return e[i];}

    // return the (i,j)-th element of a matrix
    const float& operator()(int i, int j) const {return e[j][i];}
    float& operator()(int i, int j) {return e[j][i];}
};

//---------------------------------------------------------------------------//

fmat3_cm operator+(const fmat3_cm& lhs, const fmat3_cm& rhs)
{
    return fmat3_cm(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_cm operator-(const fmat3_cm& lhs, const fmat3_cm& rhs)
{
    return fmat3_cm(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_cm operator*(float lhs, const fmat3_cm& rhs)
{
    return fmat3_cm(lhs*rhs[0], lhs*rhs[1], lhs*rhs[2]);
}

//---------------------------------------------------------------------------//

fmat3_cm operator*(const fmat3_cm& lhs, float rhs)
{
    return fmat3_cm(lhs[0]*rhs, lhs[1]*rhs, lhs[2]*rhs);
}

//---------------------------------------------------------------------------//

fmat3_cm operator/(const fmat3_cm& lhs, float rhs)
{
    return fmat3_cm(lhs[0]/rhs, lhs[1]/rhs, lhs[2]/rhs);
}

//---------------------------------------------------------------------------//

fvec3 operator*(const fmat3_cm& lhs, const fvec3& rhs)
{
	// M v = M(:,1) v + M(:,2) v + M(:,3) v
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}

//---------------------------------------------------------------------------//

fvec3 operator*(const fvec3& lhs, const fmat3_cm& rhs)
{
	//         [ v dot M(:,1) ]
	// v^T M = [ v dot M(:,2) ]
	//         [ v dot M(:,3) ]
    return fvec3(dot(lhs,rhs[0]), dot(lhs,rhs[1]), dot(lhs,rhs[2]));
}

//---------------------------------------------------------------------------//

fmat3_cm operator*(const fmat3_cm& lhs, const fmat3_cm& rhs)
{
	// A B = [ A B(:,1)  A B(:,2)  A B(:,3) ]
    return fmat3_cm(lhs*rhs[0], lhs*rhs[1], lhs*rhs[2]);
}

//---------------------------------------------------------------------------//

float det(const fmat3_cm& arg)
{
	// Laplace expansion
    return dot(arg[0],cross(arg[1],arg[2]));
}

//---------------------------------------------------------------------------//

fmat3_cm transpose(const fmat3_cm& arg)
{
    return fmat3_cm(arg(0,0), arg(1,0), arg(2,0),
                    arg(0,1), arg(1,1), arg(2,1),
                    arg(0,2), arg(1,2), arg(2,2));                 
}

//---------------------------------------------------------------------------//

fmat3_cm inv(const fmat3_cm& arg)
{
    fmat3_cm res;
	// cofactor matrix
    res[0] = cross(arg[1],arg[2]);
    res[1] = cross(arg[2],arg[0]);
    res[2] = cross(arg[0],arg[1]);
    float det = dot(arg[0],res[0]);
	// M^{-1} = cof(M)^T / det(M) = adj(M) / det(M)
    return 1.0f/det * transpose(res);
}

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& lhs, const fmat3_cm& rhs)
{
    lhs << "[[" << rhs[0][0] << "] [" << rhs[1][0] << "] [" << rhs[2][0] << "]\n"
        << " [" << rhs[0][1] << "],[" << rhs[1][1] << "],[" << rhs[2][1] << "]\n"
        << " [" << rhs[0][2] << "] [" << rhs[1][2] << "] [" << rhs[2][2] << "]]";
    return lhs;
}

//---------------------------------------------------------------------------//
