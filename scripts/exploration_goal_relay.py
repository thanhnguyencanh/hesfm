#!/usr/bin/env python3
"""
Exploration Goal Relay for HESFM

Relays exploration goals to move_base with goal management.

Author: Thanh Nguyen Canh <thanhnc@jaist.ac.jp>
Date: 2026
"""

import rospy
import numpy as np
from threading import Lock

from geometry_msgs.msg import PoseStamped, PoseArray
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib


class ExplorationGoalRelay:
    """Relay exploration goals to move_base."""
    
    def __init__(self):
        rospy.init_node('exploration_goal_relay', anonymous=False)
        
        # Parameters
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)
        self.replan_interval = rospy.get_param('~replan_interval', 10.0)
        self.timeout = rospy.get_param('~timeout', 60.0)
        
        self.lock = Lock()
        self.current_goal = None
        self.goal_time = None
        self.exploring = True
        
        # Action client
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server()
        rospy.loginfo("Connected to move_base")
        
        # Subscribers
        self.goal_sub = rospy.Subscriber(
            'exploration_goal', PoseStamped, self.goal_callback, queue_size=1)
        self.goals_sub = rospy.Subscriber(
            'exploration_goals', PoseArray, self.goals_callback, queue_size=1)
            
        # Timer for periodic checks
        self.check_timer = rospy.Timer(rospy.Duration(1.0), self.check_callback)
        
        rospy.loginfo("Exploration goal relay initialized")
        
    def goal_callback(self, msg):
        """Single best goal callback."""
        with self.lock:
            if not self.exploring:
                return
                
            # Check if significantly different from current goal
            if self.current_goal is not None:
                dx = msg.pose.position.x - self.current_goal.pose.position.x
                dy = msg.pose.position.y - self.current_goal.pose.position.y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < self.goal_tolerance:
                    return
                    
            self.send_goal(msg)
            
    def goals_callback(self, msg):
        """Multiple goals callback - use first (best) goal."""
        if len(msg.poses) > 0:
            goal = PoseStamped()
            goal.header = msg.header
            goal.pose = msg.poses[0]
            self.goal_callback(goal)
            
    def send_goal(self, pose_stamped):
        """Send goal to move_base."""
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        
        self.move_base.send_goal(
            goal,
            done_cb=self.done_callback,
            feedback_cb=self.feedback_callback)
            
        self.current_goal = pose_stamped
        self.goal_time = rospy.Time.now()
        
        rospy.loginfo(f"Sent goal: ({pose_stamped.pose.position.x:.2f}, "
                      f"{pose_stamped.pose.position.y:.2f})")
                      
    def done_callback(self, status, result):
        """Goal completion callback."""
        with self.lock:
            if status == GoalStatus.SUCCEEDED:
                rospy.loginfo("Goal reached!")
            elif status == GoalStatus.PREEMPTED:
                rospy.loginfo("Goal preempted")
            elif status == GoalStatus.ABORTED:
                rospy.logwarn("Goal aborted")
            else:
                rospy.logwarn(f"Goal finished with status: {status}")
                
            self.current_goal = None
            
    def feedback_callback(self, feedback):
        """Goal feedback callback."""
        pass  # Could track progress here
        
    def check_callback(self, event):
        """Periodic check callback."""
        with self.lock:
            if self.current_goal is None:
                return
                
            # Check for timeout
            elapsed = (rospy.Time.now() - self.goal_time).to_sec()
            if elapsed > self.timeout:
                rospy.logwarn("Goal timeout, canceling")
                self.move_base.cancel_goal()
                self.current_goal = None
                
            # Check for replan
            elif elapsed > self.replan_interval:
                state = self.move_base.get_state()
                if state == GoalStatus.ACTIVE:
                    rospy.loginfo("Replan interval reached, accepting new goals")
                    
    def stop(self):
        """Stop exploration."""
        with self.lock:
            self.exploring = False
            if self.current_goal is not None:
                self.move_base.cancel_goal()
                self.current_goal = None
            rospy.loginfo("Exploration stopped")
            
    def start(self):
        """Start exploration."""
        with self.lock:
            self.exploring = True
            rospy.loginfo("Exploration started")
            
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        node = ExplorationGoalRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
