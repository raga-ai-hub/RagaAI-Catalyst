import { Moon, Sun } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useTheme } from '@/theme/ThemeProvider'

export function ModeToggle () {
  const { setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant='ghost'
          size='icon'
          className='ml-2 bg-white/10 hover:bg-white/20 border-white/10 backdrop-blur-sm'
        >
          <Sun className='h-5 w-5 rotate-0 scale-100 transition-transform duration-200 text-white dark:-rotate-90 dark:scale-0' />
          <Moon className='absolute h-5 w-5 rotate-90 scale-0 transition-transform duration-200 text-white dark:rotate-0 dark:scale-100' />
          <span className='sr-only'>Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align='end' className='mt-2'>
        <DropdownMenuItem
          onClick={() => setTheme('light')}
          className='cursor-pointer'
        >
          <Sun className='h-4 w-4 mr-2' />
          Light
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => setTheme('dark')}
          className='cursor-pointer'
        >
          <Moon className='h-4 w-4 mr-2' />
          Dark
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => setTheme('system')}
          className='cursor-pointer'
        >
          <svg
            className='h-4 w-4 mr-2'
            fill='none'
            viewBox='0 0 24 24'
            strokeWidth={1.5}
            stroke='currentColor'
          >
            <path
              strokeLinecap='round'
              strokeLinejoin='round'
              d='M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25'
            />
          </svg>
          System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default ModeToggle
