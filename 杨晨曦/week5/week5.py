import numpy as np
import random
import sys
import matplotlib.pyplot as plt

'''
基于Kmeans算法实现类内距离排序
扩展自原始kmeans.py脚本
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

class ClusterAnalyzer:
    """聚类分析器，用于分析聚类结果并按类内距离排序"""
    
    def __init__(self, kmeans_result, centers):
        """
        初始化聚类分析器
        
        参数:
        kmeans_result: 聚类结果，是一个列表，每个元素是一个簇的点集合
        centers: 聚类中心点
        """
        self.clusters = kmeans_result
        self.centers = centers
        self.cluster_distances = self._calculate_cluster_distances()
    
    def _calculate_cluster_distances(self):
        """计算每个簇的类内距离"""
        cluster_distances = []
        
        for i, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                cluster_distances.append({
                    'cluster_id': i,
                    'intra_distance': 0,
                    'avg_distance': 0,
                    'max_distance': 0,
                    'min_distance': 0,
                    'point_count': 0
                })
                continue
                
            center = self.centers[i]
            distances = []
            
            # 计算簇内每个点到中心的距离
            for point in cluster:
                dist = self._euclidean_distance(point, center)
                distances.append(dist)
            
            # 计算类内距离统计信息
            total_distance = sum(distances)
            avg_distance = total_distance / len(distances) if distances else 0
            max_distance = max(distances) if distances else 0
            min_distance = min(distances) if distances else 0
            
            cluster_distances.append({
                'cluster_id': i,
                'intra_distance': total_distance,
                'avg_distance': avg_distance,
                'max_distance': max_distance,
                'min_distance': min_distance,
                'point_count': len(cluster)
            })
        
        return cluster_distances
    
    def _euclidean_distance(self, p1, p2):
        """计算两点间的欧氏距离"""
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)
    
    def sort_clusters_by_intra_distance(self, ascending=True):
        """
        按类内距离对簇进行排序
        
        参数:
        ascending: 是否升序排序，默认为True
        
        返回:
        排序后的簇列表和对应的距离信息
        """
        sorted_indices = sorted(
            range(len(self.cluster_distances)),
            key=lambda i: self.cluster_distances[i]['intra_distance'],
            reverse=not ascending
        )
        
        sorted_clusters = [self.clusters[i] for i in sorted_indices]
        sorted_distances = [self.cluster_distances[i] for i in sorted_indices]
        
        return sorted_clusters, sorted_distances
    
    def sort_clusters_by_avg_distance(self, ascending=True):
        """
        按平均类内距离对簇进行排序
        
        参数:
        ascending: 是否升序排序，默认为True
        
        返回:
        排序后的簇列表和对应的距离信息
        """
        sorted_indices = sorted(
            range(len(self.cluster_distances)),
            key=lambda i: self.cluster_distances[i]['avg_distance'],
            reverse=not ascending
        )
        
        sorted_clusters = [self.clusters[i] for i in sorted_indices]
        sorted_distances = [self.cluster_distances[i] for i in sorted_indices]
        
        return sorted_clusters, sorted_distances
    
    def print_cluster_analysis(self):
        """打印聚类分析结果"""
        print("聚类分析结果:")
        print("=" * 80)
        print(f"{'簇ID':<8}{'点数量':<10}{'类内总距离':<15}{'平均距离':<15}{'最大距离':<15}{'最小距离':<15}")
        print("-" * 80)
        
        for cluster_info in self.cluster_distances:
            print(f"{cluster_info['cluster_id']:<8}"
                  f"{cluster_info['point_count']:<10}"
                  f"{cluster_info['intra_distance']:<15.4f}"
                  f"{cluster_info['avg_distance']:<15.4f}"
                  f"{cluster_info['max_distance']:<15.4f}"
                  f"{cluster_info['min_distance']:<15.4f}")
    
    def plot_cluster_distances(self):
        """绘制簇距离的柱状图"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        cluster_ids = [info['cluster_id'] for info in self.cluster_distances]
        intra_distances = [info['intra_distance'] for info in self.cluster_distances]
        avg_distances = [info['avg_distance'] for info in self.cluster_distances]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 类内总距离
        ax1.bar(cluster_ids, intra_distances, color='skyblue')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Intra-cluster Distance')
        ax1.set_title('Total Intra-cluster Distance by Cluster')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 平均类内距离
        ax2.bar(cluster_ids, avg_distances, color='lightgreen')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Average Intra-cluster Distance')
        ax2.set_title('Average Intra-cluster Distance by Cluster')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('cluster_distances.png')
        plt.show()

def main():
    """主函数，演示聚类分析和排序功能"""
    # 生成随机数据
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    x = np.random.rand(100, 8)
    
    # 执行K-means聚类
    kmeans = KMeansClusterer(x, 10)
    result, centers, total_distance = kmeans.cluster()
    
    print(f"聚类完成，总类内距离: {total_distance:.4f}")
    print(f"共生成 {len(result)} 个簇")
    
    # 创建聚类分析器
    analyzer = ClusterAnalyzer(result, centers)
    
    # 打印聚类分析结果
    analyzer.print_cluster_analysis()
    
    # 按类内总距离排序
    print("\n按类内总距离升序排序:")
    print("=" * 80)
    sorted_clusters, sorted_distances = analyzer.sort_clusters_by_intra_distance(ascending=True)
    
    for i, cluster_info in enumerate(sorted_distances):
        print(f"排序后位置 {i+1}: 簇 {cluster_info['cluster_id']}, "
              f"类内总距离: {cluster_info['intra_distance']:.4f}, "
              f"点数量: {cluster_info['point_count']}")
    
    # 按平均类内距离排序
    print("\n按平均类内距离升序排序:")
    print("=" * 80)
    sorted_clusters_avg, sorted_distances_avg = analyzer.sort_clusters_by_avg_distance(ascending=True)
    
    for i, cluster_info in enumerate(sorted_distances_avg):
        print(f"排序后位置 {i+1}: 簇 {cluster_info['cluster_id']}, "
              f"平均类内距离: {cluster_info['avg_distance']:.4f}, "
              f"点数量: {cluster_info['point_count']}")
    
    # 绘制距离分布图
    analyzer.plot_cluster_distances()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
