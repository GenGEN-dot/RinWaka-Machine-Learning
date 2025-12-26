import requests
import time
import argparse
import sys
from datetime import datetime

class ConfigurableGitHubMonitor:
    def __init__(self, interval_minutes=5, check_timeout=8):
        """
        初始化监控器
        
        Args:
            interval_minutes: 监控间隔（分钟），默认5分钟
            check_timeout: 每次检查的超时时间（秒），默认8秒
        """
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.check_timeout = check_timeout
        self.consecutive_failures = 0
        self.max_retries_before_alert = 3
        
        # 要监控的GitHub端点
        self.endpoints = {
            "主页": "https://github.com",
            "API": "https://api.github.com",
            "Raw文件": "https://raw.githubusercontent.com",
            "状态页": "https://www.githubstatus.com/"
        }
        
        self.check_count = 0
        self.success_count = 0
        
    def show_notification(self, title, message, urgent=False):
        """显示Windows通知"""
        try:
            # 尝试使用win10toast
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            
            duration = 15 if urgent else 10
            
            toaster.show_toast(
                title=title,
                msg=message,
                duration=duration,
                threaded=True
            )
            return True
        except ImportError:
            # 如果未安装win10toast，尝试使用plyer
            try:
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    app_name='GitHub监控',
                    timeout=10,
                )
                return True
            except ImportError:
                # 最后使用系统弹窗
                try:
                    import ctypes
                    style = 0x30  # 警告图标
                    if urgent:
                        style = 0x10  # 错误图标
                    ctypes.windll.user32.MessageBoxW(0, message, title, style)
                    return True
                except:
                    print(f"无法显示通知: {title} - {message}")
                    return False
    
    def check_endpoint(self, name, url):
        """检查单个端点"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=self.check_timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return True, f"✅ {name}: {elapsed_ms:.0f}ms", elapsed_ms
            else:
                return False, f"❌ {name}: 状态码 {response.status_code}", None
                
        except requests.exceptions.Timeout:
            return False, f"⏰ {name}: 连接超时", None
        except requests.exceptions.ConnectionError:
            return False, f"🔌 {name}: 连接失败", None
        except Exception as e:
            return False, f"⚠️ {name}: {str(e)[:50]}", None
    
    def check_all_endpoints(self):
        """检查所有端点"""
        self.check_count += 1
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 第{self.check_count}次检查开始")
        print("-" * 60)
        
        results = []
        response_times = []
        any_failed = False
        
        for name, url in self.endpoints.items():
            success, message, elapsed = self.check_endpoint(name, url)
            results.append(message)
            
            if success and elapsed:
                response_times.append(elapsed)
            if not success:
                any_failed = True
            
            print(message)
            time.sleep(0.5)  # 稍微间隔一下，避免请求过快
        
        return any_failed, results, response_times
    
    def generate_summary(self, any_failed, results, response_times):
        """生成检查摘要"""
        if not any_failed:
            self.success_count += 1
            success_rate = (self.success_count / self.check_count) * 100
            
            if response_times:
                avg_response = sum(response_times) / len(response_times)
                summary = f"✅ 所有连接正常 (平均响应: {avg_response:.0f}ms)"
            else:
                summary = "✅ 所有连接正常"
            
            summary += f"\n成功率: {success_rate:.1f}% ({self.success_count}/{self.check_count})"
            return summary, False
        else:
            # 统计失败数量
            failed_count = sum(1 for r in results if not r.startswith("✅"))
            total_count = len(results)
            
            self.consecutive_failures += 1
            summary = f"❌ {failed_count}/{total_count} 个连接失败"
            summary += f"\n连续失败次数: {self.consecutive_failures}"
            
            # 是否需要紧急通知
            urgent = self.consecutive_failures >= self.max_retries_before_alert
            if urgent:
                summary += f"\n⚠️ 已连续失败{self.consecutive_failures}次，请立即检查！"
            
            return summary, urgent
    
    def run_check(self):
        """执行一次完整的检查"""
        any_failed, results, response_times = self.check_all_endpoints()
        summary, urgent = self.generate_summary(any_failed, results, response_times)
        
        print("-" * 60)
        print(summary)
        
        # 如果有问题，发送通知
        if any_failed:
            message = f"GitHub连接检查失败\n\n" + "\n".join(results)
            message += f"\n\n{summary}"
            message += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            title = "⚠️ 紧急：GitHub连接异常" if urgent else "⚠️ GitHub连接问题"
            self.show_notification(title, message, urgent)
        elif self.consecutive_failures > 0:
            # 刚刚恢复
            self.consecutive_failures = 0
            recovery_msg = f"GitHub连接已恢复\n\n之前的连接问题已解决"
            recovery_msg += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.show_notification("✅ GitHub连接恢复", recovery_msg)
        
        return not any_failed
    
    def show_waiting_progress(self):
        """显示等待进度"""
        total_seconds = self.interval_seconds
        interval_minutes = self.interval_minutes
        
        print(f"\n⏳ 下次检查: {interval_minutes}分钟后 (按Ctrl+C停止)...")
        
        # 每10秒更新一次进度
        for remaining in range(total_seconds, 0, -10):
            minutes_left = remaining // 60
            seconds_left = remaining % 60
            
            if remaining % 60 == 0 or remaining == total_seconds:
                if minutes_left > 0:
                    print(f"  剩余时间: {minutes_left}分{seconds_left:02d}秒", end='\r')
                else:
                    print(f"  剩余时间: {seconds_left}秒        ", end='\r')
            
            time.sleep(10)
        
        print("  开始新的检查...        ")
    
    def run_continuous_monitoring(self):
        """运行持续监控"""
        print("=" * 60)
        print("🎯 GitHub Windows监控系统")
        print("=" * 60)
        print(f"📊 监控间隔: {self.interval_minutes}分钟")
        print(f"⏱️  超时设置: {self.check_timeout}秒")
        print(f"🔍 监控端点: {len(self.endpoints)}个")
        print("=" * 60)
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                self.run_check()
                
                if self.interval_minutes > 0:
                    self.show_waiting_progress()
                else:
                    # 如果间隔为0，则只检查一次
                    print("\n⏹️ 监控完成（间隔设置为0分钟）")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 监控已手动停止")
        except Exception as e:
            error_msg = f"监控程序异常: {str(e)}"
            print(f"\n❌ {error_msg}")
            self.show_notification("❌ GitHub监控错误", error_msg)
        
        # 显示最终统计
        if self.check_count > 0:
            print("\n" + "=" * 60)
            print("📈 监控统计:")
            print(f"   总检查次数: {self.check_count}")
            print(f"   成功次数: {self.success_count}")
            
            if self.check_count > 0:
                success_rate = (self.success_count / self.check_count) * 100
                print(f"   成功率: {success_rate:.1f}%")
            
            print("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GitHub连通性监控工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python github_monitor.py              # 使用默认5分钟间隔
  python github_monitor.py -i 1         # 每分钟检查一次
  python github_monitor.py -i 10        # 每10分钟检查一次
  python github_monitor.py -i 0         # 只检查一次不循环
  python github_monitor.py -i 2 -t 5    # 每2分钟检查一次，超时5秒
        """
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=5,
        help='监控间隔（分钟），0表示只检查一次，默认5分钟'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=8,
        help='每次检查的超时时间（秒），默认8秒'
    )
    
    parser.add_argument(
        '--test-notification',
        action='store_true',
        help='测试通知系统，然后退出'
    )
    
    parser.add_argument(
        '--list-endpoints',
        action='store_true',
        help='列出所有监控的端点，然后退出'
    )
    
    return parser.parse_args()


def test_notification_system():
    """测试通知系统"""
    print("🔔 测试通知系统...")
    
    test_monitor = ConfigurableGitHubMonitor()
    
    # 测试普通通知
    print("1. 测试普通通知...")
    test_monitor.show_notification(
        "测试通知",
        "这是一个测试通知，如果你能看到这个，说明通知系统工作正常！"
    )
    
    time.sleep(2)
    
    # 测试紧急通知
    print("2. 测试紧急通知...")
    test_monitor.show_notification(
        "紧急测试",
        "这是一个紧急测试通知！",
        urgent=True
    )
    
    print("✅ 通知测试完成，请检查是否收到通知")
    time.sleep(3)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 测试通知
    if args.test_notification:
        test_notification_system()
        return
    
    # 列出端点
    if args.list_endpoints:
        monitor = ConfigurableGitHubMonitor()
        print("📡 监控端点列表:")
        for name, url in monitor.endpoints.items():
            print(f"  • {name}: {url}")
        return
    
    # 验证参数
    if args.interval < 0:
        print("❌ 错误：监控间隔不能为负数")
        sys.exit(1)
    
    if args.timeout < 1:
        print("❌ 错误：超时时间必须大于0秒")
        sys.exit(1)
    
    if args.interval == 0:
        print("🔍 单次检查模式（不循环）")
    
    # 创建并运行监控器
    monitor = ConfigurableGitHubMonitor(
        interval_minutes=args.interval,
        check_timeout=args.timeout
    )
    
    # 添加自定义端点的示例（取消注释并修改）
    # monitor.endpoints["自定义"] = "https://your-custom-endpoint.com"
    
    monitor.run_continuous_monitoring()


if __name__ == "__main__":
    # 检查是否安装了requests
    try:
        import requests
    except ImportError:
        print("❌ 未安装requests库，请先运行: pip install requests")
        print("如果需要通知功能，请运行: pip install requests win10toast plyer")
        sys.exit(1)
    
    main()