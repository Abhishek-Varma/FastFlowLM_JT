#!/usr/bin/env bash
# Fail if a Linux package ships, links or depends on a library FLM dropped for
# licensing reasons: FFmpeg and the GPL codecs distro FFmpeg pulls in (x264,
# x265, xvid, libdvdnav/libdvdread), FFTW and GNU readline.
#
# Usage: check_no_copyleft_libs.sh <package.tar.gz | package.deb>...
set -euo pipefail

# Shared-library file names (bundled files and ELF NEEDED entries).
lib_re='^lib(av(codec|format|util|filter|device)|sw(scale|resample)|postproc|x264|x265|xvidcore|dvdnav|dvdread|fftw3[a-z_]*|readline)[.]so'
# Debian binary package names (.deb Depends).
pkg_re='^(libav(codec|format|util|filter|device)|libsw(scale|resample)|libpostproc|libx26[45]|libxvidcore|libdvd(nav|read)|libfftw3|libreadline)'

status=0
for pkg in "$@"; do
    tmp=$(mktemp -d)
    case "$pkg" in
        *.deb)
            if dpkg-deb -f "$pkg" Depends | tr ',|' '\n\n' | sed 's/^ *//' | grep -E "$pkg_re"; then
                echo "::error::$pkg depends on a package listed above"
                status=1
            fi
            dpkg-deb -x "$pkg" "$tmp" ;;
        *.tar.gz)
            tar xzf "$pkg" -C "$tmp" ;;
        *)
            echo "::error::unknown package type: $pkg"
            exit 2 ;;
    esac

    if find "$tmp" \( -type f -o -type l \) -printf '%f\n' | grep -E "$lib_re"; then
        echo "::error::$pkg bundles a library listed above"
        status=1
    fi
    while IFS= read -r -d '' f; do
        if readelf -d "$f" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -E "$lib_re"; then
            echo "::error::${f#"$tmp"/} in $pkg links a library listed above"
            status=1
        fi
    done < <(find "$tmp" -type f -print0)
    rm -rf "$tmp"
    [ "$status" -eq 0 ] && echo "$pkg: no FFmpeg, GPL codec, FFTW or readline libraries"
done
exit "$status"
