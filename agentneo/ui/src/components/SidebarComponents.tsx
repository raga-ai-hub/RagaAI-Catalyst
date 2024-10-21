import React from 'react';
import { NavLink } from 'react-router-dom';
import { LucideIcon } from 'lucide-react';

interface SidebarLinkProps {
  to: string;
  icon: LucideIcon;
  text: string;
  isCollapsed: boolean;
}

export const SidebarLink: React.FC<SidebarLinkProps> = ({ to, icon: Icon, text, isCollapsed }) => (
  <li>
    <NavLink to={to} className={({ isActive }) => `flex items-center p-2 rounded-lg hover:bg-white hover:bg-opacity-20 transition-all duration-200 ${isActive ? 'bg-white bg-opacity-20' : ''}`}>
      <Icon size={20} className={isCollapsed ? 'mx-auto' : 'mr-3'} />
      {!isCollapsed && text}
    </NavLink>
  </li>
);

interface SidebarExternalLinkProps {
  href: string;
  icon: LucideIcon;
  text: string;
  isCollapsed: boolean;
}

export const SidebarExternalLink: React.FC<SidebarExternalLinkProps> = ({ href, icon: Icon, text, isCollapsed }) => (
  <li>
    <a href={href} target="_blank" rel="noopener noreferrer" className="flex items-center p-2 rounded-lg hover:bg-white hover:bg-opacity-20 transition-all duration-200">
      <Icon size={20} className={isCollapsed ? 'mx-auto' : 'mr-3'} />
      {!isCollapsed && text}
    </a>
  </li>
);