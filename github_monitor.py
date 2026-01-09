"""
GitHub连通性监控工具

使用示例:
  python github_monitor.py              # 使用默认5分钟间隔
  python github_monitor.py -i 1         # 每分钟检查一次
  python github_monitor.py -i 10        # 每10分钟检查一次
  python github_monitor.py -i 0         # 只检查一次不循环
  python github_monitor.py -i 2 -t 5    # 每2分钟检查一次，超时5秒
  python github_monitor.py --test-notification  # 测试通知系统
  python github_monitor.py --list-endpoints     # 列出监控端点

功能:
1. 监控GitHub连通性
2. 连接失败时发送精简的Windows通知
3. 自动保存日志并清理15天前的旧日志
"""

import requests
import time
import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

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
        }
        
        # 初始化统计
        self.check_count = 0
        self.success_count = 0
        
        # 创建日志目录
        self.log_dir = Path("github_monitor_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 清理旧日志
        self._cleanup_old_logs()
        
        # 创建本次运行的日志文件
        self.session_log_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.session_log = []
        
    def _cleanup_old_logs(self):
        """清理15天前的旧日志"""
        cutoff_date = datetime.now() - timedelta(days=15)
        
        for log_file in self.log_dir.glob("*.log"):
            try:
                # 从文件名中提取时间信息
                if log_file.name.startswith("session_"):
                    file_date_str = log_file.stem.split("_")[1]  # 提取日期部分
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    
                    # 如果是15天前的文件，删除它
                    if file_date < cutoff_date.date():
                        log_file.unlink()
                        print(f"🗑️ 已删除旧日志: {log_file.name}")
            except Exception as e:
                print(f"⚠️ 处理日志文件时出错 {log_file}: {e}")
    
    def _log_session(self):
        """保存本次运行的日志到文件"""
        try:
            with open(self.session_log_file, 'w', encoding='utf-8') as f:
                f.write(f"GitHub监控会话日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"监控间隔: {self.interval_minutes}分钟\n")
                f.write(f"检查超时: {self.check_timeout}秒\n")
                f.write("=" * 60 + "\n\n")
                
                for log_entry in self.session_log:
                    f.write(log_entry + "\n")
                
                # 添加统计信息
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"📈 会话统计:\n")
                f.write(f"   总检查次数: {self.check_count}\n")
                f.write(f"   成功次数: {self.success_count}\n")
                
                if self.check_count > 0:
                    success_rate = (self.success_count / self.check_count) * 100
                    f.write(f"   成功率: {success_rate:.1f}%\n")
                
            print(f"📝 日志已保存到: {self.session_log_file}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")
    
    def _add_to_session_log(self, message):
        """添加消息到会话日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.session_log.append(log_entry)
        print(log_entry)
    
    def show_notification(self, title, message, urgent=False):
        """显示Windows通知 - 精简版本"""
        # 精简通知消息，只显示关键信息
        if len(message) > 200:  # 如果消息太长，进行截断
            lines = message.split('\n')
            # 保留最重要的几行
            important_lines = []
            for line in lines:
                if '连接失败' in line or '连接超时' in line or '状态码' in line:
                    important_lines.append(line)
                if len(important_lines) >= 3:  # 最多显示3行重要信息
                    break
            
            if not important_lines:
                important_lines = lines[:3]  # 如果没有找到重要行，取前3行
            
            message = '\n'.join(important_lines)
            if len(message) > 200:
                message = message[:197] + "..."
        
        try:
            # 尝试使用win10toast
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            
            duration = 15 if urgent else 10
            
            # 修复：去掉threaded参数或设为False
            toaster.show_toast(
                title=title,
                msg=message,
                duration=duration,
                threaded=False  # 改为False避免线程问题
            )
            return True
        except Exception as e:
            print(f"win10toast通知失败，尝试其他方法: {e}")
            
            # 尝试使用系统弹窗
            try:
                import ctypes
                style = 0x40  # 信息图标
                if urgent:
                    style = 0x30  # 警告图标
                
                # 确保消息框显示在前台
                style = style | 0x10000 | 0x40000  # MB_SETFOREGROUND | MB_TOPMOST
                
                # 显示消息框
                ctypes.windll.user32.MessageBoxW(0, message, title, style)
                return True
            except Exception as e2:
                print(f"系统弹窗也失败: {e2}")
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
        self._add_to_session_log(f"第{self.check_count}次检查开始")
        self._add_to_session_log("-" * 60)
        
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
            
            self._add_to_session_log(message)
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
        
        self._add_to_session_log("-" * 60)
        self._add_to_session_log(summary)
        
        # 如果有问题，发送通知
        if any_failed:
            # 精简通知消息 - 只显示失败的连接
            failed_results = [r for r in results if not r.startswith("✅")]
            success_results = [r for r in results if r.startswith("✅")]
            
            # 构建精简消息
            if failed_results:
                message = f"GitHub连接失败!\n\n" + "\n".join(failed_results)
                if success_results:
                    message += f"\n\n正常连接:\n" + "\n".join(success_results[:2])  # 最多显示2个正常连接
                    if len(success_results) > 2:
                        message += f"\n...还有{len(success_results)-2}个连接正常"
                
                # 确保消息不会太长
                if len(message) > 500:
                    message = message[:497] + "..."
                
                message += f"\n\n时间: {datetime.now().strftime('%H:%M:%S')}"
                
                title = "⚠️ 紧急：连接失败！请检查连接！" if urgent else "⚠️ 连接失败！请检查连接！"
                self.show_notification(title, message, urgent)
        elif self.consecutive_failures > 0:
            # 刚刚恢复
            self.consecutive_failures = 0
            recovery_msg = f"GitHub连接已恢复!\n之前的连接问题已解决\n时间: {datetime.now().strftime('%H:%M:%S')}"
            self.show_notification("✅ GitHub连接恢复", recovery_msg)
        
        return not any_failed
    
    def show_waiting_progress(self):
        """显示等待进度"""
        total_seconds = self.interval_seconds
        interval_minutes = self.interval_minutes
        
        self._add_to_session_log(f"\n⏳ 下次检查: {interval_minutes}分钟后 (按Ctrl+C停止)...")
        
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
        
        self._add_to_session_log("GitHub监控会话开始")
        self._add_to_session_log(f"监控间隔: {self.interval_minutes}分钟")
        self._add_to_session_log(f"检查超时: {self.check_timeout}秒")
        
        try:
            while True:
                self.run_check()
                
                if self.interval_minutes > 0:
                    self.show_waiting_progress()
                else:
                    # 如果间隔为0，则只检查一次
                    self._add_to_session_log("监控完成（间隔设置为0分钟）")
                    print("\n⏹️ 监控完成（间隔设置为0分钟）")
                    break
                    
        except KeyboardInterrupt:
            self._add_to_session_log("监控已手动停止")
            print("\n\n👋 监控已手动停止")
        except Exception as e:
            error_msg = f"监控程序异常: {str(e)}"
            self._add_to_session_log(f"错误: {error_msg}")
            print(f"\n❌ {error_msg}")
            self.show_notification("❌ GitHub监控错误", error_msg)
        finally:
            # 保存日志
            self._log_session()
            
            # 显示最终统计
            if self.check_count > 0:
                print("\n" + "=" * 60)
                print("📈 监控统计:")
                print(f"   总检查次数: {self.check_count}")
                print(f"   成功次数: {self.success_count}")
                
                if self.check_count > 0:
                    success_rate = (self.success_count / self.check_count) * 100
                    print(f"   成功率: {success_rate:.1f}%")
                
                print(f"   日志文件: {self.session_log_file}")
                print("=" * 60)


def print_usage_examples():
    """打印使用示例"""
    print("使用示例:")
    print("  python github_monitor.py              # 使用默认5分钟间隔")
    print("  python github_monitor.py -i 1         # 每分钟检查一次")
    print("  python github_monitor.py -i 10        # 每10分钟检查一次")
    print("  python github_monitor.py -i 0         # 只检查一次不循环")
    print("  python github_monitor.py -i 2 -t 5    # 每2分钟检查一次，超时5秒")
    print("  python github_monitor.py --test-notification  # 测试通知系统")
    print("  python github_monitor.py --list-endpoints     # 列出监控端点")
    print()


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
    
    # 测试精简通知
    print("1. 测试精简通知...")
    test_monitor.show_notification(
        "测试通知",
        "这是一个测试通知，消息内容比较短，应该能完全显示。"
    )
    
    time.sleep(2)
    
    # 测试长消息（会被自动精简）
    print("2. 测试长消息通知...")
    long_message = "GitHub连接失败!\n\n"
    long_message += "⏰ 主页: 连接超时\n"
    long_message += "✅ API: 542ms\n"
    long_message += "✅ Raw文件: 321ms\n"
    long_message += "⚠️ 状态页: 状态码 404\n"
    long_message += "⏰ 另一个端点: 连接超时\n"
    long_message += "✅ 又一个端点: 123ms\n"
    long_message += "\n时间: 12:34:56"
    
    test_monitor.show_notification(
        "⚠️ 连接测试",
        long_message
    )
    
    print("✅ 通知测试完成，请检查是否收到通知")
    time.sleep(3)


def main():
    """主函数"""
    # 打印使用示例
    print_usage_examples()
    
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
        print("如果需要通知功能，请运行: pip install win10toast")
        sys.exit(1)
    
    main()