///*
// * ==============================================================
// * RMP_mult.c  Compute a matrix vector multiplication for Max-Pooling Matching
// *
// * Minsu CHO
// * 8 Sep 2013
// * =============================================================
// */

//#include "mex.h"

///*
// * The mex function computes the reweighted max-pooling
// */
///*
//    INPUT: *prhs[] contains: 1. Affinity Matrix ( matM ).
//                             2. Initial vectorized permutation matrix ( x )
//                             3. Groups of matches (G_val,G_row,G_col) to query the Affinity Matrix

//    OUTPUT: updated permutation matrix ( x' )
//*/


//void mexFunction(int nlhs, mxArray *plhs[],
//                 int nrhs, const mxArray *prhs[])
//{
//    int i, j, k, g, ge;
//    int mrows, ncols, ngroups;

//    /* sparse matrix */
//    mwIndex *G_row, *G_col;
//    bool *G_val;

//    double *x;
//    double *y;
//    double *matM;

//    double val, vmax;
//    int imax;
//    bool bSame;

//    if (nrhs != 3)
//    {
//        mexErrMsgTxt("3 inputs required.");
//    }
//    else if (nlhs > 1)
//    {
//        mexErrMsgTxt("Too many output arguments");
//    }

//    /* The first input must be a nonsparse matrix. */
//    mrows = mxGetM(prhs[0]); // rows of Affinity matrix
//    ncols = mxGetN(prhs[0]); // cols of Affinity matrix
//    if (mxIsSparse(prhs[0]) ||
//        !mxIsDouble(prhs[0]) ||
//        mxIsComplex(prhs[0]))
//    {
//        mexErrMsgTxt("Input must be a noncomplex nonsparse matrix.");
//    }

//    /* The second input must be a vector. */
//    /* X */
//    if (mxGetM(prhs[1])*mxGetN(prhs[1]) != ncols ||
//        mxIsSparse(prhs[1]) || !mxIsDouble(prhs[1]))
//    {
//        mexErrMsgTxt("Invalid vector.");
//    }

//    /* The third input must be a nonsparse matrix. */
//    ngroups = mxGetN(prhs[2]);
//    if (!mxIsSparse(prhs[2]) || mxGetM(prhs[2]) != mrows ||
//        !mxIsLogical(prhs[2]))
//    {
//        mexErrMsgTxt("Invalid group matrix.");
//    }

//    /* Get the sparse matrix */
//    /* Groups of features to retrive the similarity values*/
//    G_val = mxGetLogicals(prhs[2]);
//    G_row = mxGetIr(prhs[2]);
//    G_col = mxGetJc(prhs[2]);

//    /* Get the nonsparse matrix */
//    /* Get the Affinity Matrix */
//    matM = mxGetPr(prhs[0]);

//    /* Get the vector x */
//    x = mxGetPr(prhs[1]);

//    plhs[0] = mxCreateDoubleMatrix(mrows,1,mxREAL);
//    /* Initialize the permutation verctor x */
//    y = mxGetPr(plhs[0]);
//    for (j = 0; j < mrows; j++)
//        y[j] = 0;

//    // sum(j=0 -> N_i)
//    for (j = 0; j < mrows; j++) // rows of Affinity matrix
//    {
//        // max-pooling (b=0 -> N_a)
//        for (g = 0; g < ngroups; ++g) // number of groups n1 x n2
//        {
//            vmax = -9999;
//            bSame = false;

//            for (ge = G_col[g]; ge < G_col[g+1]; ++ge)
//            {
//                if ( G_row[ge] == j )
//                {
//                    bSame = true;
//                    break;
//                }

//                // look for the edge similarity
//                val = x[G_row[ge]] * matM[ j + mrows*G_row[ge] ];
//                // max-pooling
//                if ( val > vmax ){
//                    vmax = val;
//                    imax = G_row[ge];
//                }
//            }

//            if ( !bSame )
//            {
//                y[j] += vmax*vmax;
//            }
//        }

//    }
//}

//% get k-nearest neighbors
//NNi = P1(KNN_ij{i,1}, :); sizeNNi = size(NNi, 1);
//NNj = P1(KNN_ij{j,1}, :); sizeNNj = size(NNj, 1);
//NNa = P2(KNN_ab{a,1}, :); sizeNNa = size(NNa, 1);
//NNb = P2(KNN_ab{b,1}, :); sizeNNb = size(NNb, 1);

