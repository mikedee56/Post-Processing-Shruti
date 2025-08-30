"""
Expert Notification System for Story 3.2.1: Expert Review Queue System

This module provides comprehensive notification capabilities for alerting expert linguists
when they have pending review tickets in the Expert Review Queue. Supports multiple
notification channels including email, Slack, SMS, and in-app notifications.

Part of Epic 3: Semantic Refinement & QA Framework
Story 3.2.1: Expert Review Queue System - Task 5
"""

import asyncio
import smtplib
import json
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import requests
from abc import ABC, abstractmethod

# Import internal modules
from qa_module.expert_review_queue import ReviewTicket, TicketPriority, TicketStatus
from utils.logger_config import get_logger


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationConfig:
    """Configuration for the notification system."""
    enabled_channels: Set[NotificationChannel] = field(default_factory=lambda: {NotificationChannel.EMAIL})
    email_config: Dict[str, Any] = field(default_factory=dict)
    slack_config: Dict[str, Any] = field(default_factory=dict)
    sms_config: Dict[str, Any] = field(default_factory=dict)
    webhook_config: Dict[str, Any] = field(default_factory=dict)
    notification_frequency: int = 300  # seconds between batch notifications
    max_notifications_per_batch: int = 10
    enable_digest_mode: bool = True
    digest_schedule: str = "hourly"  # hourly, daily
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    enable_quiet_hours: bool = True


@dataclass
class ExpertContact:
    """Contact information for an expert."""
    expert_id: str
    name: str
    email: Optional[str] = None
    slack_user_id: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    preferred_channels: Set[NotificationChannel] = field(default_factory=lambda: {NotificationChannel.EMAIL})
    notification_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationMessage:
    """A notification message to be sent."""
    recipient_expert_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    subject: str
    content: str
    ticket_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivery_status: str = "pending"


@dataclass
class NotificationTemplate:
    """Template for generating notifications."""
    name: str
    subject_template: str
    content_template: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: Set[NotificationChannel] = field(default_factory=lambda: {NotificationChannel.EMAIL})


class NotificationChannel_ABC(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    async def send_notification(self, message: NotificationMessage, contact: ExpertContact) -> bool:
        """Send a notification message through this channel."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the configuration for this channel."""
        pass


class EmailNotificationChannel(NotificationChannel_ABC):
    """Email notification channel implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_address = config.get('from_address', 'noreply@example.com')
        self.use_tls = config.get('use_tls', True)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email configuration."""
        required_fields = ['smtp_server', 'from_address']
        return all(field in config for field in required_fields)
    
    async def send_notification(self, message: NotificationMessage, contact: ExpertContact) -> bool:
        """Send email notification."""
        if not contact.email:
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = contact.email
            msg['Subject'] = message.subject
            
            # Add content
            msg.attach(MIMEText(message.content, 'html' if '<' in message.content else 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel_ABC):
    """Slack notification channel implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.bot_token = config.get('bot_token')
        self.channel = config.get('default_channel', '#general')
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Slack configuration."""
        return bool(config.get('webhook_url') or config.get('bot_token'))
    
    async def send_notification(self, message: NotificationMessage, contact: ExpertContact) -> bool:
        """Send Slack notification."""
        if not (self.webhook_url or self.bot_token):
            return False
        
        try:
            # Format message for Slack
            slack_message = {
                "text": f"{message.subject}\n{message.content}",
                "username": "Expert Review System",
                "icon_emoji": ":bell:",
            }
            
            if contact.slack_user_id:
                slack_message["channel"] = f"@{contact.slack_user_id}"
            else:
                slack_message["channel"] = self.channel
            
            # Send via webhook
            if self.webhook_url:
                response = requests.post(self.webhook_url, json=slack_message, timeout=10)
                return response.status_code == 200
            
            # TODO: Implement bot token method if needed
            return False
            
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel_ABC):
    """Generic webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 30)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook configuration."""
        return bool(config.get('webhook_url'))
    
    async def send_notification(self, message: NotificationMessage, contact: ExpertContact) -> bool:
        """Send webhook notification."""
        if not self.webhook_url:
            return False
        
        try:
            payload = {
                'expert_id': message.recipient_expert_id,
                'priority': message.priority.value,
                'subject': message.subject,
                'content': message.content,
                'ticket_ids': message.ticket_ids,
                'contact': {
                    'name': contact.name,
                    'email': contact.email,
                    'timezone': contact.timezone
                },
                'timestamp': message.created_at.isoformat(),
                'metadata': message.metadata
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")
            return False


class ExpertNotificationSystem:
    """
    Comprehensive notification system for expert alerts.
    
    This system handles notification delivery for expert review queue tickets,
    supporting multiple channels, batching, scheduling, and delivery tracking.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """Initialize the notification system."""
        self.config = config or NotificationConfig()
        self.logger = get_logger(__name__)
        
        # Initialize storage
        self.expert_contacts: Dict[str, ExpertContact] = {}
        self.notification_templates: Dict[str, NotificationTemplate] = {}
        self.pending_notifications: List[NotificationMessage] = []
        self.sent_notifications: List[NotificationMessage] = []
        
        # Initialize notification channels
        self.channels: Dict[NotificationChannel, NotificationChannel_ABC] = {}
        self._initialize_channels()
        
        # Initialize templates
        self._initialize_default_templates()
        
        # Background task management
        self._notification_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info("ExpertNotificationSystem initialized")
    
    def _initialize_channels(self):
        """Initialize notification channels based on configuration."""
        if NotificationChannel.EMAIL in self.config.enabled_channels:
            if self.config.email_config:
                email_channel = EmailNotificationChannel(self.config.email_config)
                if email_channel.validate_config(self.config.email_config):
                    self.channels[NotificationChannel.EMAIL] = email_channel
                    self.logger.info("Email notification channel initialized")
                else:
                    self.logger.warning("Invalid email configuration, channel disabled")
        
        if NotificationChannel.SLACK in self.config.enabled_channels:
            if self.config.slack_config:
                slack_channel = SlackNotificationChannel(self.config.slack_config)
                if slack_channel.validate_config(self.config.slack_config):
                    self.channels[NotificationChannel.SLACK] = slack_channel
                    self.logger.info("Slack notification channel initialized")
                else:
                    self.logger.warning("Invalid Slack configuration, channel disabled")
        
        if NotificationChannel.WEBHOOK in self.config.enabled_channels:
            if self.config.webhook_config:
                webhook_channel = WebhookNotificationChannel(self.config.webhook_config)
                if webhook_channel.validate_config(self.config.webhook_config):
                    self.channels[NotificationChannel.WEBHOOK] = webhook_channel
                    self.logger.info("Webhook notification channel initialized")
                else:
                    self.logger.warning("Invalid webhook configuration, channel disabled")
    
    def _initialize_default_templates(self):
        """Initialize default notification templates."""
        # New ticket assigned template
        self.notification_templates['ticket_assigned'] = NotificationTemplate(
            name="ticket_assigned",
            subject_template="New Review Ticket Assigned: {ticket_id}",
            content_template="""
            Dear {expert_name},
            
            A new review ticket has been assigned to you:
            
            Ticket ID: {ticket_id}
            Priority: {priority}
            Issue Type: {issue_type}
            Created: {created_at}
            
            Please review this ticket at your earliest convenience.
            
            Segment Text: {segment_text}
            
            Best regards,
            Expert Review System
            """,
            priority=NotificationPriority.NORMAL,
            channels={NotificationChannel.EMAIL, NotificationChannel.SLACK}
        )
        
        # Urgent ticket template
        self.notification_templates['urgent_ticket'] = NotificationTemplate(
            name="urgent_ticket",
            subject_template="🚨 URGENT: Critical Review Required - {ticket_id}",
            content_template="""
            Dear {expert_name},
            
            An urgent review ticket requires your immediate attention:
            
            Ticket ID: {ticket_id}
            Priority: HIGH/URGENT
            Issue Type: {issue_type}
            Created: {created_at}
            
            This ticket requires review within 2 hours.
            
            Segment Text: {segment_text}
            Complexity Score: {complexity_score}
            
            Please prioritize this review.
            
            Best regards,
            Expert Review System
            """,
            priority=NotificationPriority.URGENT,
            channels={NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK}
        )
        
        # Daily digest template
        self.notification_templates['daily_digest'] = NotificationTemplate(
            name="daily_digest",
            subject_template="Daily Review Summary - {pending_count} Pending Tickets",
            content_template="""
            Dear {expert_name},
            
            Here's your daily review summary:
            
            Pending Tickets: {pending_count}
            Completed Today: {completed_count}
            Average Review Time: {avg_review_time}
            
            High Priority Tickets:
            {high_priority_tickets}
            
            Normal Priority Tickets:
            {normal_priority_tickets}
            
            Please log in to the review dashboard to continue your work.
            
            Best regards,
            Expert Review System
            """,
            priority=NotificationPriority.LOW,
            channels={NotificationChannel.EMAIL}
        )
    
    def register_expert(self, contact: ExpertContact):
        """Register an expert for notifications."""
        self.expert_contacts[contact.expert_id] = contact
        self.logger.info(f"Registered expert {contact.name} ({contact.expert_id}) for notifications")
    
    def unregister_expert(self, expert_id: str):
        """Unregister an expert from notifications."""
        if expert_id in self.expert_contacts:
            del self.expert_contacts[expert_id]
            self.logger.info(f"Unregistered expert {expert_id} from notifications")
    
    async def notify_ticket_assigned(self, ticket: ReviewTicket, expert_id: str, **kwargs) -> bool:
        """Send notification when a ticket is assigned to an expert."""
        if expert_id not in self.expert_contacts:
            self.logger.warning(f"Expert {expert_id} not registered for notifications")
            return False
        
        contact = self.expert_contacts[expert_id]
        template = self.notification_templates['ticket_assigned']
        
        # Determine priority based on ticket
        if ticket.priority == TicketPriority.URGENT:
            template = self.notification_templates['urgent_ticket']
        
        # Format message
        subject = template.subject_template.format(
            ticket_id=ticket.ticket_id,
            **kwargs
        )
        
        content = template.content_template.format(
            expert_name=contact.name,
            ticket_id=ticket.ticket_id,
            priority=ticket.priority.value.upper(),
            issue_type=ticket.issue_type,
            created_at=ticket.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            segment_text=ticket.segment_text[:200] + "..." if len(ticket.segment_text) > 200 else ticket.segment_text,
            complexity_score=ticket.metadata.get('complexity_score', 'N/A'),
            **kwargs
        )
        
        # Create notification message for each preferred channel
        success_count = 0
        for channel in contact.preferred_channels:
            if channel in template.channels and channel in self.channels:
                notification = NotificationMessage(
                    recipient_expert_id=expert_id,
                    channel=channel,
                    priority=template.priority,
                    subject=subject,
                    content=content,
                    ticket_ids=[ticket.ticket_id],
                    metadata={'ticket_id': ticket.ticket_id, 'action': 'assigned'}
                )
                
                success = await self._send_notification(notification, contact)
                if success:
                    success_count += 1
        
        return success_count > 0
    
    async def notify_ticket_overdue(self, ticket: ReviewTicket, expert_id: str) -> bool:
        """Send notification for overdue tickets."""
        if expert_id not in self.expert_contacts:
            return False
        
        contact = self.expert_contacts[expert_id]
        
        subject = f"⏰ Overdue Review: {ticket.ticket_id}"
        content = f"""
        Dear {contact.name},
        
        The following review ticket is overdue:
        
        Ticket ID: {ticket.ticket_id}
        Assigned: {ticket.assigned_at.strftime('%Y-%m-%d %H:%M:%S') if ticket.assigned_at else 'N/A'}
        Due: {ticket.due_date.strftime('%Y-%m-%d %H:%M:%S') if ticket.due_date else 'N/A'}
        Priority: {ticket.priority.value.upper()}
        
        Please complete this review as soon as possible.
        
        Best regards,
        Expert Review System
        """
        
        # Send via all preferred channels
        success_count = 0
        for channel in contact.preferred_channels:
            if channel in self.channels:
                notification = NotificationMessage(
                    recipient_expert_id=expert_id,
                    channel=channel,
                    priority=NotificationPriority.HIGH,
                    subject=subject,
                    content=content,
                    ticket_ids=[ticket.ticket_id],
                    metadata={'ticket_id': ticket.ticket_id, 'action': 'overdue'}
                )
                
                success = await self._send_notification(notification, contact)
                if success:
                    success_count += 1
        
        return success_count > 0
    
    async def send_daily_digest(self, expert_id: str, digest_data: Dict[str, Any]) -> bool:
        """Send daily digest to an expert."""
        if expert_id not in self.expert_contacts:
            return False
        
        contact = self.expert_contacts[expert_id]
        template = self.notification_templates['daily_digest']
        
        # Format digest content
        subject = template.subject_template.format(
            pending_count=digest_data.get('pending_count', 0)
        )
        
        content = template.content_template.format(
            expert_name=contact.name,
            pending_count=digest_data.get('pending_count', 0),
            completed_count=digest_data.get('completed_count', 0),
            avg_review_time=digest_data.get('avg_review_time', 'N/A'),
            high_priority_tickets=digest_data.get('high_priority_summary', 'None'),
            normal_priority_tickets=digest_data.get('normal_priority_summary', 'None')
        )
        
        # Send digest (typically only via email)
        notification = NotificationMessage(
            recipient_expert_id=expert_id,
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.LOW,
            subject=subject,
            content=content,
            metadata={'action': 'daily_digest'}
        )
        
        return await self._send_notification(notification, contact)
    
    async def _send_notification(self, message: NotificationMessage, contact: ExpertContact) -> bool:
        """Send a single notification message."""
        if message.channel not in self.channels:
            self.logger.warning(f"Channel {message.channel.value} not available")
            return False
        
        # Check quiet hours
        if self._is_quiet_hours(contact.timezone) and message.priority != NotificationPriority.URGENT:
            # Schedule for later
            message.scheduled_at = self._get_next_active_time(contact.timezone)
            self.pending_notifications.append(message)
            return True
        
        # Send immediately
        channel = self.channels[message.channel]
        success = await channel.send_notification(message, contact)
        
        if success:
            message.sent_at = datetime.utcnow()
            message.delivery_status = "sent"
            self.sent_notifications.append(message)
            self.logger.info(f"Notification sent to {contact.expert_id} via {message.channel.value}")
        else:
            message.delivery_status = "failed"
            self.logger.error(f"Failed to send notification to {contact.expert_id} via {message.channel.value}")
        
        return success
    
    def _is_quiet_hours(self, timezone: str) -> bool:
        """Check if it's currently quiet hours for the given timezone."""
        if not self.config.enable_quiet_hours:
            return False
        
        # Simplified implementation - assumes UTC for now
        current_time = datetime.utcnow().time()
        quiet_start = datetime.strptime(self.config.quiet_hours_start, '%H:%M').time()
        quiet_end = datetime.strptime(self.config.quiet_hours_end, '%H:%M').time()
        
        if quiet_start <= quiet_end:
            return quiet_start <= current_time <= quiet_end
        else:  # Quiet hours span midnight
            return current_time >= quiet_start or current_time <= quiet_end
    
    def _get_next_active_time(self, timezone: str) -> datetime:
        """Get the next time outside quiet hours."""
        # Simplified implementation
        next_active = datetime.utcnow().replace(
            hour=int(self.config.quiet_hours_end.split(':')[0]),
            minute=int(self.config.quiet_hours_end.split(':')[1]),
            second=0,
            microsecond=0
        )
        
        if next_active <= datetime.utcnow():
            next_active += timedelta(days=1)
        
        return next_active
    
    async def start_background_processing(self):
        """Start background task for processing notifications."""
        if self._running:
            return
        
        self._running = True
        self._notification_task = asyncio.create_task(self._notification_processor())
        self.logger.info("Background notification processing started")
    
    async def stop_background_processing(self):
        """Stop background notification processing."""
        self._running = False
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Background notification processing stopped")
    
    async def _notification_processor(self):
        """Background task to process pending notifications."""
        while self._running:
            try:
                # Process pending notifications
                current_time = datetime.utcnow()
                to_send = []
                remaining = []
                
                for notification in self.pending_notifications:
                    if notification.scheduled_at and notification.scheduled_at <= current_time:
                        to_send.append(notification)
                    else:
                        remaining.append(notification)
                
                self.pending_notifications = remaining
                
                # Send scheduled notifications
                for notification in to_send:
                    if notification.recipient_expert_id in self.expert_contacts:
                        contact = self.expert_contacts[notification.recipient_expert_id]
                        await self._send_notification(notification, contact)
                
                # Wait before next processing cycle
                await asyncio.sleep(self.config.notification_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in notification processor: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        total_sent = len(self.sent_notifications)
        sent_by_channel = {}
        sent_by_priority = {}
        
        for notification in self.sent_notifications:
            channel = notification.channel.value
            priority = notification.priority.value
            sent_by_channel[channel] = sent_by_channel.get(channel, 0) + 1
            sent_by_priority[priority] = sent_by_priority.get(priority, 0) + 1
        
        return {
            'total_notifications_sent': total_sent,
            'pending_notifications': len(self.pending_notifications),
            'registered_experts': len(self.expert_contacts),
            'active_channels': list(self.channels.keys()),
            'notifications_by_channel': sent_by_channel,
            'notifications_by_priority': sent_by_priority,
            'background_processing_active': self._running
        }


def create_expert_notification_system(config: Optional[Dict[str, Any]] = None) -> ExpertNotificationSystem:
    """Factory function to create an ExpertNotificationSystem with configuration."""
    if config:
        notification_config = NotificationConfig(
            enabled_channels=set(NotificationChannel(ch) for ch in config.get('enabled_channels', ['email'])),
            email_config=config.get('email', {}),
            slack_config=config.get('slack', {}),
            webhook_config=config.get('webhook', {}),
            notification_frequency=config.get('notification_frequency', 300),
            enable_digest_mode=config.get('enable_digest_mode', True),
            enable_quiet_hours=config.get('enable_quiet_hours', True),
            quiet_hours_start=config.get('quiet_hours_start', '22:00'),
            quiet_hours_end=config.get('quiet_hours_end', '08:00')
        )
    else:
        notification_config = NotificationConfig()
    
    return ExpertNotificationSystem(notification_config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_notification_system():
        """Test the notification system."""
        # Create test configuration
        test_config = {
            'enabled_channels': ['email'],
            'email': {
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'from_address': 'test@example.com'
            }
        }
        
        # Initialize system
        notification_system = create_expert_notification_system(test_config)
        
        # Register test expert
        expert = ExpertContact(
            expert_id="expert_001",
            name="Dr. Sanskrit Expert",
            email="expert@example.com",
            preferred_channels={NotificationChannel.EMAIL}
        )
        notification_system.register_expert(expert)
        
        # Create test ticket
        from qa_module.expert_review_queue import ReviewTicket, TicketPriority
        test_ticket = ReviewTicket(
            ticket_id="test_001",
            segment_text="Test segment requiring expert review",
            issue_type="sanskrit_terminology",
            priority=TicketPriority.NORMAL,
            complexity_score=0.75,
            metadata={'test': True}
        )
        
        # Test notification
        success = await notification_system.notify_ticket_assigned(test_ticket, "expert_001")
        print(f"Notification sent: {success}")
        
        # Get stats
        stats = notification_system.get_notification_stats()
        print(f"Notification stats: {stats}")
    
    # Run test
    asyncio.run(test_notification_system())