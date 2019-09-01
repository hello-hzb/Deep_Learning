#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<stdio.h>
// https://www.oschina.net/code/snippet_1986028_53912
typedef double(*Function)(double);

typedef struct {
    double eta;
    double momentum;
    int szLayer;        //层数
    int *layer;         //每层的节点数
    Function act;       //激活函数
    Function actdiff;   //激活函数导数
    double **weights;   //节点权值
    double **preWeights;//前一时刻节点权值
    double **delta;     //误差
    double **theta;     //阈值
    double **preTheta;  //前一时刻阈值
    double **output;    //每层输出
}BPAnn;

//矩阵的乘积
void MatXMat(double mat1[], double mat2[], double output[], int row, int column, int lcolrrow)
{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j)
        {
            int pos = column * i + j;
            output[pos] = 0;
            for (int k = 0; k < lcolrrow; ++k)
                output[pos] += mat1[lcolrrow * i + k] * mat2[column * k + j];
        }
}

//生成-1.0~1.0随机双精度浮点数
double drand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    return *(double *)&lvalue - 3;
}

//创建神经网络，其实只是定义了参数的个数申请了内存空间而已
BPAnn* CreateBPAnn(double eta, double momentum, int layer[],int szLayer,Function act,Function actdiff)
{
    int lastIndex = szLayer - 1;

    BPAnn *pBPAnn = (BPAnn *)malloc(sizeof(BPAnn));                     //定义一个结构体用来保存
    pBPAnn->layer = (int *)malloc(sizeof(int)*szLayer);
    pBPAnn->weights = (double **)malloc(sizeof(double *)*lastIndex);
    pBPAnn->preWeights = (double **)malloc(sizeof(double *)*lastIndex);
    pBPAnn->theta = (double **)malloc(sizeof(double *)*lastIndex);      //应该是偏置
    pBPAnn->preTheta = (double **)malloc(sizeof(double *)*lastIndex);
    pBPAnn->delta = (double **)malloc(sizeof(double *)*lastIndex);      //误差
    pBPAnn->output = (double **)malloc(sizeof(double *)*szLayer);

    //参数初始化
    for (int szWeight, i = 0; i < lastIndex; ++i)
    {
        szWeight = layer[i] * layer[i + 1];
        pBPAnn->weights[i] = (double *)malloc(sizeof(double)*szWeight);
        pBPAnn->preWeights[i] = (double *)malloc(sizeof(double)*szWeight);
        pBPAnn->theta[i] = (double *)malloc(sizeof(double)*layer[i + 1]);
        pBPAnn->preTheta[i] = (double *)malloc(sizeof(double)*layer[i + 1]);
        pBPAnn->delta[i] = (double *)malloc(sizeof(double)*layer[i + 1]);
        pBPAnn->output[i] = (double *)malloc(sizeof(double)*layer[i]);
        pBPAnn->layer[i] = layer[i];

        for (int j = 0; j < szWeight; ++j) {
            pBPAnn->weights[i][j] = drand();
            pBPAnn->preWeights[i][j] = 0;
        }
        for (int j = 0; j < layer[i + 1]; ++j)
        {
            pBPAnn->theta[i][j] = drand();
            pBPAnn->preTheta[i][j] = 0;
        }
    }
    pBPAnn->output[lastIndex] = (double *)malloc(sizeof(double)*layer[lastIndex]);
    pBPAnn->layer[lastIndex] = layer[lastIndex];
    pBPAnn->eta = eta;
    pBPAnn->momentum = momentum;
    pBPAnn->szLayer = szLayer;
    pBPAnn->act = act;
    pBPAnn->actdiff = actdiff;
    return pBPAnn;
}

//销毁神经网络占用的内存
int DestroyBPAnn(BPAnn *pBPAnn)
{
    if (!pBPAnn) return 0;
    int lastIndex = pBPAnn->szLayer - 1;
    for (int i = 0; i < lastIndex; ++i)
    {
        free(pBPAnn->weights[i]);
        free(pBPAnn->preWeights[i]);
        free(pBPAnn->theta[i]);
        free(pBPAnn->preTheta[i]);
        free(pBPAnn->delta[i]);
        free(pBPAnn->output[i]);
    }
    free(pBPAnn->output[lastIndex]);
    free(pBPAnn->layer);
    free(pBPAnn->weights);
    free(pBPAnn->preWeights);
    free(pBPAnn->theta);
    free(pBPAnn->preTheta);
    free(pBPAnn->delta);
    free(pBPAnn->output);
    free(pBPAnn);
    return 1;
}

static void LoadInput(double input[],BPAnn *pBPAnn)
{
    for (int i = 0; i < pBPAnn->layer[0]; ++i)
        pBPAnn->output[0][i] = input[i];
}


//target为真实值，delta为对激活函数求偏导数，然后乘于误差
static void LoadTarget(double target[], BPAnn *pBPAnn)
{
    int lastIndex = pBPAnn->szLayer - 1;
    double *delta = pBPAnn->delta[lastIndex - 1];
    double *output = pBPAnn->output[lastIndex];
    for (int i = 0; i < pBPAnn->layer[lastIndex]; ++i)
        delta[i] = pBPAnn->actdiff(output[i])*(target[i] - output[i]);
}
//每一层前向计算过程
static void Forward(BPAnn *pBPAnn)
{
    int lastIndex = pBPAnn->szLayer - 1;  //最后一次层的索引
    int *layer = pBPAnn->layer;           //每一层的节点数
    double **weights = pBPAnn->weights;   //网络的权重
    double **output = pBPAnn->output;     //每一层网络输出
    double **theta = pBPAnn->theta;       //网络的偏置
    Function act = pBPAnn->act;           //激活函数
    for (int i = 0; i < lastIndex; ++i)
    {
        MatXMat(output[i], weights[i], output[i + 1], 1, layer[i + 1], layer[i]); //前一层的输出乘于权重
        for (int j = 0; j < layer[i + 1]; ++j)
            output[i + 1][j] = act(output[i + 1][j] + theta[i][j]);               //加上偏置并经过激活函数作用
    }
}
//计算误差，反向逐层计算误差
static void CalculateDelta(BPAnn *pBPAnn)
{
    int lastIndex = pBPAnn->szLayer - 1;
    int *layer = pBPAnn->layer;
    double **weights = pBPAnn->weights;
    double **output = pBPAnn->output;
    double **delta = pBPAnn->delta;
    Function actdiff = pBPAnn->actdiff;
    for (int i = lastIndex-1; i > 0; --i)
    {
        MatXMat(weights[i], delta[i], delta[i - 1], layer[i], 1, layer[i + 1]);
        for (int j = 0; j < layer[i]; ++j)
            delta[i - 1][j] *= actdiff(output[i][j]);
    }
}

//更新参数，包括权重和偏置
static void AdjustWeights(BPAnn *pBPAnn)
{
    int lastIndex = pBPAnn->szLayer - 1;
    int *layer = pBPAnn->layer;
    double **weights = pBPAnn->weights;
    double **output = pBPAnn->output;
    double **delta = pBPAnn->delta;
    double **preWeights = pBPAnn->preWeights;
    double **theta = pBPAnn->theta;
    double **preTheta = pBPAnn->preTheta;
    double momentum = pBPAnn->momentum;         //这个好像是作用在旧的权重上面
    double eta = pBPAnn->eta;                   //这个应该是学习率
    for (int i = 0; i < lastIndex; ++i)
    {
        for (int j = 0; j < layer[i]; ++j)
            for (int k = 0; k < layer[i + 1]; ++k)
            {
                int pos = j*layer[i + 1] + k;
                preWeights[i][pos] = momentum * preWeights[i][pos] + eta * delta[i][k] * output[i][j];
                weights[i][pos] += preWeights[i][pos];
            }

        for (int j = 0; j < layer[i + 1]; ++j)
        {
            preTheta[i][j] = momentum*preTheta[i][j] + eta*delta[i][j];
            theta[i][j] += preTheta[i][j];
        }
    }
}

void Train(double input[], double target[],BPAnn *pBPAnn)
{
    LoadInput(input, pBPAnn);
    Forward(pBPAnn);
    LoadTarget(target,pBPAnn);
    CalculateDelta(pBPAnn);
    AdjustWeights(pBPAnn);
}
//预测过程计算
void Predict(double input[],double output[],BPAnn *pBPAnn)
{
    int lastIndex = pBPAnn->szLayer - 1;
    LoadInput(input, pBPAnn);
    Forward(pBPAnn);
    double *result = pBPAnn->output[lastIndex];
    for (int i = 0; i < pBPAnn->layer[lastIndex]; ++i)
        output[i] = result[i];
}

double Sigmod(double x)
{
    return 1 / (1 + exp(-x));
}

double SigmodDiff(double y)
{
    return y*(1 - y);
}

double Leaky_Relu(double x, double alpha)
{
    return (x>0)?(x):(alpha*x);
}

double Leaky_Relu_Diff(double y, double alpha)
{
    return (y>0)?(1):(alpha);
}

//将数转化成2进制数
static void ToBinary(unsigned x, unsigned n,double output[])
{
    for (unsigned i = 0, j = x; i < n; ++i, j >>= 1)
        output[i] = j & 1;
}
//将二进制数转化成十进制数
static unsigned FromBinary(double output[],unsigned n)
{
    int result = 0;
    for (int i = n - 1; i >= 0; --i)
        result = result << 1 | (output[i] > 0.5) ;//对输出结果四舍五入，并通过二进制转换为数
    return result;
}

//使用神经网络进行异或运算，输入为2个0~32767之间的数，前15节点为第1个数二进制，后15节点为第2个数二进制，输出为异或结果的二进制
int main()
{
    int layer[] = {30,48,15};
    //1.学习率，2.动量，3.每一层的节点数,4.层数，5.激活函数，6.激活函数的导数
    BPAnn *bp = CreateBPAnn(0.25, 0.9, layer, 3, Sigmod, SigmodDiff);
    double input[30], output[15];
    int error = 0;
    printf("训练进度：   ");
    for (int j = 0; j < 100; ++j) {
        for (int i = 0; i < 20000; ++i)
        {
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x^y, 15, output);         //x^y表示x和y的异或
            Train(input, output, bp);
        }
        printf("\b\b\b%02d%%", j + 1);
    }
    printf("\n训练完毕\n");
    for (int i = 0; i < 2000; ++i)
    {
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        Predict(input, output, bp);
        unsigned result = FromBinary(output, 15); //输出每一个点的输出结果
        error += result != (x^y);                 //计算误差
        printf("%u^%u=%u\tpredict=%u\n", x, y, x^y, result);
    }
    printf("error=%d\n", error);
    DestroyBPAnn(bp);
    system("pause");
    return 0;
}