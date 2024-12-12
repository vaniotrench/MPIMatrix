#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

void generateMatrix(int rows, int cols, vector<int>& matrix) {
    srand(time(0));
    for (int i = 0; i < rows * cols; i++) {
        matrix.push_back(rand() % 10); 
    }
}

void displayMatrix(const vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

void readMatrixFromFile(const string& filename, vector<int>& matrix, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    file >> rows >> cols;
    matrix.resize(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file >> matrix[i * cols + j];
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A_rows, A_cols, B_rows, B_cols;
    vector<int> A, B, C;

    if (rank == 0) {
        int choice;
        cout << "Matrix Multiplication Program\n";
        cout << "1. Read matrices from file\n";
        cout << "2. Generate matrices randomly\n";
        cout << "Enter your choice: ";
        cin >> choice;

        if (choice == 1) {
            string fileA, fileB;
            cout << "Enter filename for Matrix A: ";
            cin >> fileA;
            readMatrixFromFile(fileA, A, A_rows, A_cols);

            cout << "Enter filename for Matrix B: ";
            cin >> fileB;
            readMatrixFromFile(fileB, B, B_rows, B_cols);
        }
        else if (choice == 2) {
            cout << "Enter rows and columns for Matrix A: ";
            cin >> A_rows >> A_cols;
            cout << "Enter rows and columns for Matrix B: ";
            cin >> B_rows >> B_cols;

            if (A_cols != B_rows) {
                cout << "Matrix multiplication not possible. A's columns must equal B's rows." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            generateMatrix(A_rows, A_cols, A);
            generateMatrix(B_rows, B_cols, B);

            cout << "Matrix A:" << endl;
            displayMatrix(A, A_rows, A_cols);

            cout << "Matrix B:" << endl;
            displayMatrix(B, B_rows, B_cols);
        }
        else {
            cout << "Invalid choice. Exiting program." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        A.resize(A_rows * A_cols);
        B.resize(B_rows * B_cols);
    }

    MPI_Bcast(A.data(), A_rows * A_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), B_rows * B_cols, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsPerProcess = A_rows / size;
    int extraRows = A_rows % size;

    int startRow = rank * rowsPerProcess + min(rank, extraRows);
    int rowsAssigned = rowsPerProcess + (rank < extraRows ? 1 : 0);

    vector<int> subA(rowsAssigned * A_cols);

    for (int i = 0; i < rowsAssigned; i++) {
        for (int j = 0; j < A_cols; j++) {
            subA[i * A_cols + j] = A[(startRow + i) * A_cols + j];
        }
    }

    vector<int> subC(rowsAssigned * B_cols, 0);
    for (int i = 0; i < rowsAssigned; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                subC[i * B_cols + j] += subA[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }

    if (rank == 0) {
        C.resize(A_rows * B_cols); 
    }
    vector<int> recvCounts(size);
    vector<int> displs(size);

    for (int i = 0; i < size; i++) {
        recvCounts[i] = rowsPerProcess * B_cols + (i < extraRows ? B_cols : 0);
        displs[i] = (i > 0) ? displs[i - 1] + recvCounts[i - 1] : 0;
    }

    MPI_Gatherv(subC.data(), subC.size(), MPI_INT, C.data(),
        recvCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Resultant Matrix C:" << endl;
        displayMatrix(C, A_rows, B_cols);
    }

    MPI_Finalize();
    return 0;
}
