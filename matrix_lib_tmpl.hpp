#pragma once
#include <ostream>
#include <cmath>

template<typename T>
concept has_add = requires (T a, T b) { a+b; };

template<typename T>
concept has_sub = requires (T a, T b) { a-b; };

template<typename T>
concept has_mul = requires (T a, T b) { a*b; };

template<typename T>
concept has_div = requires (T a, T b) { a/b; };

template<typename T>
concept has_add_mul = requires (T a, T b) {a+b;a*b;};

template<typename T, typename S>
concept has_add_pow = requires (T a, T b, S c) {a+b;std::pow(S(a),c);};

template<typename T>
concept has_sub_mul = requires (T a, T b) {a-b;a*b;};

template<typename T>
concept has_os_out = requires (std::ostream& os, T a) {os<<a;};

template<typename T>
concept has_add_sub_mul = requires (T a, T b) {a+b;a-b;a*b;};

template<typename T>
concept has_add_sub_mul_div = requires (T a, T b) {a+b;a-b;a*b;a/b;};

// overload this for complex numbers to return the complex conjugate
template<typename T>
T conj(const T& a) {return a;}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//                vector with reasonable amounts of template code            //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

// a vector that can be interpreted as both a column- and a row-vector
// convenient for when we do not want to switch things around
template<typename T, int n>
struct vec
{
    static_assert(n>0, "Vectors must have a positive integer dimension.");
    
    T e[n];
    
    vec() {}

    // initialize to the same value
    vec(const T& s) 
    {
        for (int i=0; i<n; ++i)
            e[i] = s;
    }

    // initialise with the arguments
    template<typename ...Ts>
    vec(const Ts& ...es) : e{static_cast<T>(es)...} 
    {
        static_assert(sizeof...(Ts)==n, "Too many arguments in vec<T,n> ctor.");
    }

    const T& operator[](int i) const {return e[i];}
    T& operator[](int i) {return e[i];}
    const T& operator()(int i) const {return e[i];}
    T& operator()(int i) {return e[i];}
};

//---------------------------------------------------------------------------//

template<has_add T, int n>
vec<T,n> operator+(const vec<T,n>& lhs, const vec<T,n>& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] + rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_sub T, int n>
vec<T,n> operator-(const vec<T,n>& lhs, const vec<T,n>& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] - rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_mul T, int n>
vec<T,n> operator*(const T& lhs, const vec<T,n>& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_mul T, int n>
vec<T,n> operator*(const vec<T,n>& lhs, const T& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] * rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_div T, int n>
vec<T,n> operator/(const vec<T,n>& lhs, const T& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] / rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_mul T, int n>
vec<T,n> hmul(const vec<T,n>& lhs, const vec<T,n>& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] * rhs[i];
    return res;
} // elementwise/Hadamard product

//---------------------------------------------------------------------------//

template<has_div T, int n>
vec<T,n> hdiv(const vec<T,n>& lhs, const vec<T,n>& rhs)
{
    vec<T,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] / rhs[i];
    return res;
} // elementwise/Hadamard division

//---------------------------------------------------------------------------//

template<has_add_mul T, int n>
T dot(const vec<T,n>& lhs, const vec<T,n>& rhs)
{
    T res = lhs[0]*rhs[0];
    for (int i=1; i<n; ++i)
        res = res + conj(lhs[i]) * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<typename T>
concept has_dot = requires (const T& a, const T& b) { dot(a,b); };

//---------------------------------------------------------------------------//

template<has_add T, int n>
T norm1(const vec<T,n>& arg)
{
    T res = std::abs(arg[0]);
    for (int i=1; i<n; ++i)
        res = res + std::abs(arg[i]);
    return res;
}

//---------------------------------------------------------------------------//

auto norm2(const has_dot auto& arg)
{
    return std::sqrt(dot(arg,arg));
}

//---------------------------------------------------------------------------//

template<has_add T, int n>
T norminf(const vec<T,n>& arg)
{
    T res = std::abs(arg[0]);
    for (int i=1; i<n; ++i)
        res = std::max(res, std::abs(arg[i]));
    return res;
}

//---------------------------------------------------------------------------//

template<typename T, int n, typename S> requires has_add_pow<T,S>
auto norm(const vec<T,n>& arg, S p)
{
    auto res = std::pow(std::abs(arg[0]),p);
    for (int i=1; i<n; ++i)
        res = res + std::pow(std::abs(arg[i]),p);
    return pow(res,S(1)/p);
}

//---------------------------------------------------------------------------//

template<has_sub_mul T>
vec<T,3> cross(const vec<T, 3>& lhs, const vec<T,3>& rhs)
{
    return vec<T,3>(lhs[1]*rhs[2]-lhs[2]*rhs[1],
                    lhs[2]*rhs[0]-lhs[0]*rhs[2],
                    lhs[0]*rhs[1]-lhs[1]*rhs[0]);
}

//---------------------------------------------------------------------------//

template<has_add_mul T, int n>
vec<T, n> proj(const vec<T,n>& onto, const vec<T,n>& arg)
{
    return dot(onto, arg)/dot(onto,onto) * onto;
}

//---------------------------------------------------------------------------//

template<has_os_out T, int n>
std::ostream& operator<<(std::ostream& lhs, const vec<T,n>& rhs)
{
    lhs << "[";
    for (int i=0; i<n; ++i)
    {
        lhs << rhs[i];
        if (i<n-1)
            lhs << ",";
    }
    lhs << "]";
    return lhs;
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//            row-major matrix with reasonable amounts of template code      //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

// row-major mxn matrix
template<typename T, int m, int n>
struct mat_rm
{
    static_assert(m>0 && n>0, "Matrix dimensions must be positive integers.");

    union
    {
        vec<vec<T,n>,m> rows;
        T e[m][n];
    };
    
    mat_rm(){}

    template<typename...Ts>
    mat_rm(Ts...es) : e{static_cast<T>(es)...} {static_assert(sizeof...(Ts)==m*n, "Invalid number of arguments.");} 

    vec<T,n>& operator[](int i) {return rows[i];}
    const vec<T,n>& operator[](int i) const {return rows[i];}

    T& operator()(int i, int j) {return e[i][j];}
    const T& operator()(int i, int j) const {return e[i][j];}
};

//---------------------------------------------------------------------------//

template<has_add T, int m, int n>
mat_rm<T,m,n> operator+(const mat_rm<T,m,n>& lhs, const mat_rm<T,m,n>& rhs)
{
    return lhs.rows + rhs.rows;
}

//---------------------------------------------------------------------------//

template<has_sub T, int m, int n>
mat_rm<T,m,n> operator-(const mat_rm<T,m,n>& lhs, const mat_rm<T,m,n>& rhs)
{
    return lhs.rows - rhs.rows;
}

//---------------------------------------------------------------------------//

template<has_mul T, int m, int n>
mat_rm<T,m,n> operator*(const T& lhs, const mat_rm<T,m,n>& rhs)
{
    mat_rm<T,m,n> res;
    for (int i=0; i<m; ++i)
        res[i] = lhs * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_mul T, int m, int n>
mat_rm<T,m,n> operator*(const mat_rm<T,m,n>& lhs, const T& rhs)
{
    mat_rm<T,m,n> res;
    for (int i=0; i<m; ++i)
        res[i] = lhs[i] * rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_div T, int m, int n>
mat_rm<T,m,n> operator/(const mat_rm<T,m,n>& lhs, const T& rhs)
{
    mat_rm<T,m,n> res;
    for (int i=0; i<m; ++i)
        res[i] = lhs[i] / rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_os_out T, int m, int n>
std::ostream& operator<<(std::ostream& lhs, const mat_rm<T,m,n>& rhs)
{
    lhs << "[";
    for (int i=0; i<n; ++i)
    {
        if (i>0)
            lhs <<" ";
        lhs << rhs[i];
        if (i<n-1)
            lhs << ",\n";
    }
    lhs << "]";
    return lhs;
}

//---------------------------------------------------------------------------//

// row-major matrix times column vector M v
template<has_add_mul T, int m, int n, int p>
vec<T,m> operator*(const mat_rm<T,m,n>& lhs, const vec<T,p>& rhs)
{
    static_assert(n==p, "Inner dimensions must match in matrix-vector multiplication.");
    vec<T,m> res;
    //       [ M(1,:) dot v ]
    // M v = [        ...   ]
    //       [ M(m,:) dot v ]
    //
    for (int i=0; i<m; ++i)
        res[i] = dot(lhs[i],rhs);
    return res;
}

//---------------------------------------------------------------------------//

// row vector times row-major matrix v^T M
template<has_add_mul T, int p, int m, int n>
vec<T,n> operator*(const vec<T,p>& lhs, const mat_rm<T,m,n>& rhs)
{
    static_assert(p==m, "Inner dimensions must match in vector-matrix multiplication.");   
    // v^T M = v(1) M(1,:) + ... + v(m) M(m,:) 
    auto res = lhs[0]*rhs[0];     
    for (int i=1; i<m; ++i)
        res = res + lhs[i] * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

// row-major matrix times row-major matrix
template<has_add_mul T, int m, int n, int p, int q>
mat_rm<T,m,q> operator*(const mat_rm<T,m,n>& lhs, const mat_rm<T,p,q>& rhs)
{
    static_assert(n==p, "Inner dimensions must match in matrix-vector multiplication.");
    mat_rm<T,m,q> res;
    //       [ A(1,:) B ]
    // A B = [    ...   ]
    //       [ A(m,:) B ]
    for (int i=0; i<m; ++i)
        res[i] = lhs[i] * rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<typename T, int m, int n>
mat_rm<T,n,m> transpose(const mat_rm<T,m,n>& arg)
{
    mat_rm<T,n,m> res;
    for (int j=0; j<n; ++j)
        for (int i=0; i<m; ++i)
            res(j,i) = arg(i,j);
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul T, int m, int n>
T det(const mat_rm<T,m,n>& arg)
{
    static_assert(m==n, "The determinant is defined only for square matrices.");
    // Laplace expansion
    T res = T(0);
    bool sgn_swap = false;
    for (int i=0; i<m; ++i, sgn_swap=!sgn_swap)
        if (!sgn_swap)
            res = res + arg[i][0] * det(matrmric0(arg,i));
        else
            res = res - arg[i][0] * det(matrmric0(arg,i));
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul T>
T det(const mat_rm<T,2,2>& arg)
{
    return arg(0,0)*arg(1,1)-arg(0,1)*arg(1,0);
}

//---------------------------------------------------------------------------//

template<typename T>
T det(const mat_rm<T,1,1>& arg)
{
    return arg(0,0);
}

//---------------------------------------------------------------------------//

// removes the r_idx-th row and the zero-th column of row-major matrix
template<typename T, int m, int n>
mat_rm<T,m-1,n-1> matrmric0(const mat_rm<T,m,n>& arg, int r_idx)
{
    static_assert(m>1, "There must be at least 2 rows.");
    static_assert(n>1, "There must be at least 2 columns.");
    mat_rm<T,m-1,n-1> res;
    for (int i=0; i<r_idx; ++i)
        res[i] = vecrme0(arg[i]);
    for (int i=r_idx; i<m-1; ++i)
        res[i] = vecrme0(arg[1+i]);
    return res;
}

//---------------------------------------------------------------------------//

// removes the first element of a vector
template<typename T, int n>
vec<T,n-1> vecrme0(const vec<T,n>& arg)
{
    static_assert(n>1, "There must be at least 2 elements.");
    vec<T,n-1> res;
    for (int i=0; i<n-1; ++i)
        res[i] = arg[1+i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul_div T, int m, int n>
mat_rm<T,n,m> inv(const mat_rm<T,m,n>& arg)
{
    static_assert(m==n, "The matrix must be square to be invertible.");

    mat_rm<T,n,m> res;
    // compute the adjugate matrix
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            if ((i+j)%2==0)
                res(i,j) = det(submat_cross(arg,j,i));
            else
                res(i,j) = -det(submat_cross(arg,j,i));
    // compute the determinant
    T det = T(0);
    for (int i=0; i<m; ++i)
        det = det + arg(i,0) * res(0, i);

    return res / det;
}

//---------------------------------------------------------------------------//

template<has_div T>
mat_rm<T,1,1> inv(const mat_rm<T,1,1>& arg)
{
    return T(1)/arg(0,0);
}

//---------------------------------------------------------------------------//

// crosses out the r_idx-th row and c_idx-th column and returns the resulting submatrix
template<typename T, int m, int n>
mat_rm<T,m-1,n-1> submat_cross(const mat_rm<T,m,n>& arg, int r_idx, int c_idx)
{
    static_assert(m>1, "There must be at least 2 rows.");
    static_assert(n>1, "There must be at least 2 columns.");

    mat_rm<T,m-1,n-1> res;
    // upper part
    for (int i=0; i<r_idx; ++i)
    {
        for (int j=0; j<c_idx; ++j)
            res(i,j) = arg(i,j);
        for (int j=c_idx; j<n-1; ++j)
            res(i,j) = arg(i,1+j);    
    }   

    // lower part
    for (int i=r_idx; i<m-1; ++i)
    {
        for (int j=0; j<c_idx; ++j)
            res(i,j) = arg(1+i,j);
        for (int j=c_idx; j<n-1; ++j)
            res(i,j) = arg(1+i,1+j);    
    }  

    return res; 
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//                                                                           //
//            column-major matrix with reasonable amounts of template code   //
//                                                                           //
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

// column-major mxn matrix
template<typename T, int m, int n>
struct mat_cm
{
    static_assert(m>0 && n>0, "Matrix dimensions must be positive integers.");

    union
    {
        vec<vec<T,m>,n> cols;
        T e[n][m];
    };

    mat_cm() {}
    
    template<typename...Ts>
    mat_cm(Ts...es) 
    {
        static_assert(sizeof...(Ts)==m*n, "Invalid number of arguments.");

        // putting in es... in the correct locations results in too much template ugliness
        // tranposing in place is also unpleasant: 
        // https://en.wikipedia.org/wiki/In-place_matrix_transposition#Non-square_matrices:_Following_the_cycles
        
        // thus I use the stupidest approach known to man
        T c[n*m] = {static_cast<T>(es)...};
        for (int j=0; j<n; ++j)
            for (int i=0; i<m; ++i)
                e[j][i] = c[i*n+j];
    } 

    vec<T,n>& operator[](int j) {return cols[j];}
    const vec<T,n>& operator[](int j) const {return cols[j];}

    T& operator()(int i, int j) {return e[j][i];}
    const T& operator()(int i, int j) const {return e[j][i];}
};

//---------------------------------------------------------------------------//

template<has_add T, int m, int n>
mat_cm<T,m,n> operator+(const mat_cm<T,m,n>& lhs, const mat_cm<T,m,n>& rhs)
{
    return lhs.cols + rhs.cols;
}

//---------------------------------------------------------------------------//

template<has_sub T, int m, int n>
mat_cm<T,m,n> operator-(const mat_cm<T,m,n>& lhs, const mat_cm<T,m,n>& rhs)
{
    return lhs.cols - rhs.cols;
}

//---------------------------------------------------------------------------//

template<has_mul T, int m, int n>
mat_cm<T,m,n> operator*(const T& lhs, const mat_cm<T,m,n>& rhs)
{
    mat_cm<T,m,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

template<has_mul T, int m, int n>
mat_cm<T,m,n> operator*(const mat_cm<T,m,n>& lhs, const T& rhs)
{
    mat_cm<T,m,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] * rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_div T, int m, int n>
mat_cm<T,m,n> operator/(const mat_cm<T,m,n>& lhs, const T& rhs)
{
    mat_cm<T,m,n> res;
    for (int i=0; i<n; ++i)
        res[i] = lhs[i] / rhs;
    return res;
}

//---------------------------------------------------------------------------//

template<has_os_out T, int m, int n>
std::ostream& operator<<(std::ostream& lhs, const mat_cm<T,m,n>& rhs)
{
    lhs << "[";
    for (int i=0; i<n; ++i)
    {
        if (i>0)
            lhs << " ";
        for (int j=0; j<m; ++j)
        {
            lhs << "[" << rhs(i,j) << "]";
            if (j<m-1)
                if (i==n/2)
                    lhs << ",";
                else
                    lhs << " ";
        }
        if (i<n-1)
            lhs << "\n";
    }
    lhs << "]";
    return lhs;
}

//---------------------------------------------------------------------------//

// column-major matrix times column vector M v
template<has_add_mul T, int m, int n, int p>
vec<T,m> operator*(const mat_cm<T,m,n>& lhs, const vec<T,p>& rhs)
{
    static_assert(n==p, "Inner dimensions must match in matrix-vector multiplication.");
    // M v = M(:,1) v(1) + ... + M(:,n) v(n)
    vec<T,m> res = lhs[0] * rhs[0];
    for (int i=1; i<n; ++i)
        res = res + lhs[i] * rhs[i];
    return res;
}

//---------------------------------------------------------------------------//

// row vector times column-major matrix v^T M
template<has_add_mul T, int p, int m, int n>
vec<T,n> operator*(const vec<T,p>& lhs, const mat_cm<T,m,n>& rhs)
{
    static_assert(p==m, "Inner dimensions must match in vector-matrix multiplication.");   
    // v^T M = [ v dot M(:,1) ... v dot M(:,n) ]
    vec<T,n> res;   
    for (int j=0; j<n; ++j)
        res[j] = dot(lhs,rhs[j]);
    return res;
}

//---------------------------------------------------------------------------//

// column-major matrix times column-major matrix
template<has_add_mul T, int m, int p, int q, int n>
mat_cm<T,m,n> operator*(const mat_cm<T,m,p>& lhs, const mat_cm<T,q,n>& rhs)
{
    static_assert(p==q, "Inner dimensions must match in matrix-vector multiplication.");
    mat_cm<T,m,n> res;
    // A B = [A B(:,1) ... A B(:,n)]
    for (int j=0; j<n; ++j)
        res[j] = lhs * rhs[j];
    return res;
}

//---------------------------------------------------------------------------//

template<typename T, int m, int n>
mat_cm<T,n,m> transpose(const mat_cm<T,m,n>& arg)
{
    mat_cm<T,n,m> res;
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
            res(j,i) = arg(i,j);
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul T, int m, int n>
T det(const mat_cm<T,m,n>& arg)
{
    static_assert(m==n, "The determinant is defined only for square matrices.");
    // Laplace expansion
    T res = T(0);
    bool sgn_swap = false;
    for (int j=0; j<n; ++j, sgn_swap=!sgn_swap)
        if (!sgn_swap)
            res = res + arg[j][0] * det(matrmr0cj(arg,j));
        else
            res = res - arg[j][0] * det(matrmr0cj(arg,j));
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul T>
T det(const mat_cm<T,2,2>& arg)
{
    return arg(0,0)*arg(1,1)-arg(0,1)*arg(1,0);
}

//---------------------------------------------------------------------------//

template<typename T>
T det(const mat_cm<T,1,1>& arg)
{
    return arg(0,0);
}

//---------------------------------------------------------------------------//

// removes the zero-th row and the c_idx-th column in a column-major matrix
template<typename T, int m, int n>
mat_cm<T,m-1,n-1> matrmr0cj(const mat_cm<T,m,n>& arg, int c_idx)
{
    static_assert(m>1, "There must be at least 2 rows.");
    static_assert(n>1, "There must be at least 2 columns.");
    mat_cm<T,m-1,n-1> res;
    for (int j=0; j<c_idx; ++j)
        res[j] = vecrme0(arg[j]);
    for (int j=c_idx; j<n-1; ++j)
        res[j] = vecrme0(arg[1+j]);
    return res;
}

//---------------------------------------------------------------------------//

template<has_add_sub_mul_div T, int m, int n>
mat_cm<T,n,m> inv(const mat_cm<T,m,n>& arg)
{
    static_assert(m==n, "The matrix must be square to be invertible.");

    mat_cm<T,n,m> res;
    // compute the adjugate matrix
    for (int j=0; j<n; ++j)
        for (int i=0; i<m; ++i)
            if ((i+j)%2==0)
                res(i,j) = det(submat_cross(arg,j,i));
            else
                res(i,j) = -det(submat_cross(arg,j,i));
    // compute the determinant
    T det = T(0);
    for (int j=0; j<n; ++j)
        det = det + arg(0,j) * res(j, 0);

    // M^{-1} = adj(M) / det(M)
    return res / det;
}

//---------------------------------------------------------------------------//

template<has_div T>
mat_cm<T,1,1> inv(const mat_cm<T,1,1>& arg)
{
    return T(1)/arg(0,0);
}

//---------------------------------------------------------------------------//

// crosses out the r_idx-th row and c_idx-th column and returns the resulting submatrix
template<typename T, int m, int n>
mat_cm<T,m-1,n-1> submat_cross(const mat_cm<T,m,n>& arg, int r_idx, int c_idx)
{
    static_assert(m>1, "There must be at least 2 rows.");
    static_assert(n>1, "There must be at least 2 columns.");

    mat_cm<T,m-1,n-1> res;
    // left part
    for (int j=0; j<c_idx; ++j)
    {
        for (int i=0; i<r_idx; ++i)
            res(i,j) = arg(i,j);
        for (int i=r_idx; i<m-1; ++i)
            res(i,j) = arg(1+i,j);    
    }   

    // right part
    for (int j=c_idx; j<n-1; ++j)
    {
        for (int i=0; i<r_idx; ++i)
            res(i,j) = arg(i,1+j);
        for (int i=r_idx; i<m-1; ++i)
            res(i,j) = arg(1+i,1+j);    
    }  

    return res; 
}

//---------------------------------------------------------------------------//