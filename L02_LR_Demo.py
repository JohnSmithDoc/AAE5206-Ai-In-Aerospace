import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_data():

    # 用于系统交互操作
    import os

    # 获取当前工作目录（Current Working Directory）
    cwd = os.getcwd()
    print(cwd)

    data = pd.read_csv('./L02_LR_Data/train.csv')

    # 地上居住面积
    x = data['GrLivArea']
    # 房屋的销售价格
    y = data['SalePrice']

    #pre-processing
    # 将 x 标准化为均值为 0、标准差为 1 的数据
    x = (x - x.mean()) / x.std()   ## Nx1

    # 在标准化后的x前添加一列全为1的列：
    #   np.ones(x.shape[0])：生成一个长度为N的全1数组。x.shape[0]获取x的行数
    #   np.c_[]：将全1数组和标准化后的x按列拼接。结果是一个形状为Nx2 的数组，
    #   第一列全为1（用于线性回归中的截距项），第二列是标准化后的x。

    x = np.c_[np.ones(x.shape[0]),x]  ##  Nx2

    return x, y

def gradient_descent(x, y):
    N = y.size  # No. of data points

    learning_rate = 0.01  # Step size
    iterations = 2000  # No. of iterations
    epsilon = 0.0001

    np.random.seed(123)

    # 这行代码生成一个包含两个随机数的一维数组（向量），这些随机数在 [0, 1) 区间内均匀分布。np.random.rand(2) 返回一个形状为 (2,) 的数组
    # 注意这里w只有两个元素哦
    w = np.random.rand(2) # step 1
    all_avg_err = []
    all_w = [w]

    for i in range(iterations):
        ## forward pass
        ## 计算x和w的点积，注意这里就是曲线拟合时我们假设是线性拟合y = wx + b，而且是两个未知数，我们就是求w 和 b
        ## 但是我们将x的一列全都变为了1,此时就可以化简为 y = w^T * x，注意这里的w有两个元素哦，一个是w,另一个就是b
        prediction = np.dot(x, w)
        errors = prediction - y
        # 利用已知的数据计算平均误差，
        avg_err = 1/N * np.dot(errors.T, errors) # step 2

        # 如果误差很小，则直接退出
        if np.dot(errors.T, errors)< epsilon: break
        all_avg_err.append(avg_err)

        ## 注意loss function 是 1/𝑁 * 误差的平方和,
        ## update w     (y^bar -(wT x + b))^2  --> 2(y^bar - (Wt)...)
        ## 注意y^bar是已知的，认为是常数，x也是已知的，也可以认为是常数，w才是变量
        ## 这里是误差对w求导，然后利用梯度下降哦， (y^bar - (wT x))^2对w求导得 L‘(w) = 2(y^bar - wTx)* (-xT)
        ## y^bar -wTx = -errors，所以 L'(w) = -2 * errors * -x^T = 2 * errors * x^T
        ## 然后因为loss function实际是有1/N的，所以变为 2/N * errors * x^T
        ## 然后利用此时的梯度乘以步长，来更新w，这样便可以迭代进行梯度下降
        w = w - learning_rate * (2/N) * np.dot(x.T, errors)
        all_w.append(w)

    return all_w, all_avg_err

def show_err(all_avg_err):
    plt.title('Errors')
    plt.xlabel('No. of iterations')
    plt.ylabel('Err')
    plt.plot(all_avg_err)
    plt.show()

def show_w(x, y, all_w, all_avg_err):
    # Set the plot up,
    fig = plt.figure()
    ax = plt.axes()
    plt.title('Sale Price vs Living Area')
    plt.xlabel('Living Area in square feet (normalised)')
    plt.ylabel('Sale Price ($)')
    plt.scatter(x[:, 1], y, color='red')
    line, = ax.plot([], [], lw=2)

    # 在坐标位置 (-1, 700000) 处创建一个空的文本注释，后续会动态变动这个注释内容
    annotation = ax.text(-1, 700000, '')
    annotation.set_animated(True)
    plt.close()

    def init():
        # 初始化一个空的线
        line.set_data([], [])
        annotation.set_text('')
        return line, annotation

    # animation function.  This is called sequentially
    def animate(i):
        # 从-5 ~ 20生成 1000个x
        x = np.linspace(-5, 20, 1000)
        # 利用上述x计算对应的y
        y = all_w[i][1] * x + all_w[i][0]
        # 应为x和y均有多个值，则可以生成一条直线
        line.set_data(x, y)
        annotation.set_text('err = %.2f e10' % (all_avg_err[i] / 10000000000))
        return line, annotation

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=0, blit=True)
    anim.save('animation.gif', writer='imagemagick', fps=30)
    print('animation saved!')




def demo():
    x, y = load_data()

    all_w, all_avg_err = gradient_descent(x, y)

    # 最后一个w就是迭代出来的最优的参数w1 和 w2， w1其实是b, w2是w y = wx + b = W^Tx
    w = all_w[-1]
    print("Estimated w1, w2: {:.2f}, {:.2f}".format(w[0], w[1]))

    # show_err(all_avg_err)
    show_w(x, y, all_w, all_avg_err)

################################
if __name__ == "__main__":
    demo()