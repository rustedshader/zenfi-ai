import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Routes that require authentication
const protectedRoutes = ["/"];

// Routes that should redirect to home if user is authenticated
const authRoutes = ["/login", "/signup"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("auth-token")?.value;

  // Debug logging
  console.log(`[Middleware] Path: ${pathname}, Token exists: ${!!token}`);

  // Check if user is authenticated
  const isAuthenticated = !!token;

  // If user is authenticated and trying to access auth routes, redirect to home
  if (isAuthenticated && authRoutes.includes(pathname)) {
    console.log(`[Middleware] Redirecting authenticated user from ${pathname} to /`);
    return NextResponse.redirect(new URL("/", request.url));
  }

  // If user is not authenticated and trying to access protected routes, redirect to login
  if (!isAuthenticated && protectedRoutes.includes(pathname)) {
    console.log(`[Middleware] Redirecting unauthenticated user from ${pathname} to /login`);
    return NextResponse.redirect(new URL("/login", request.url));
  }

  console.log(`[Middleware] Allowing access to ${pathname}`);
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except API routes, static files, and images
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
