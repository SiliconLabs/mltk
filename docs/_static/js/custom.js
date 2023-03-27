
window.SURVEY_URL = 'https://www.surveymonkey.com/r/8KCP7P5';

const MINUTE = 60;
const DAY = 24*60*60;

window.SHOW_SURVEY_AFTER_SECONDS = 10*MINUTE; // Show the survey after 10min of activity
window.INITIAL_IGNORED_SURVEY_TIMEOUT = 15*DAY; // Reshow the survey after 15 days the first time it was ignored
window.ADDITIONAL_IGNORED_SURVEY_TIMEOUT = 25*DAY; // Reshow the survey after 25 days each additional time the survey is ignored
window.RESHOW_SURVEY_TIMEOUT = 35*DAY; // Reshow the survey every 35 days after it was successfully completed


window.dataLayer = window.dataLayer || [];
window.loadingSurvey = false;
window.showingSurvey = false;
window.tookSurvey = localStorage.surveyUrl === window.SURVEY_URL


// Open external links in a new tab
$(document).ready(function () {
    addScrollToTopButton();
    determineIfCookieConsentRequired();
    $('a[href^="http://"], a[href^="https://"]').not('a[class*=internal]').attr('target', '_blank');
});


function gtag() {
    dataLayer.push(arguments);
}

function initialiseGoogleAnalytics() {
    console.log('Loading google analytics');
    gtag('js', new Date());
    gtag('config', gTrackingId, {'anonymize_ip': true});
}

function determineIfCookieConsentRequired() {
    const EU_TIMEZONES = [
        'Europe/Vienna',
        'Europe/Brussels',
        'Europe/Sofia',
        'Europe/Zagreb',
        'Asia/Famagusta',
        'Asia/Nicosia',
        'Europe/Prague',
        'Europe/Copenhagen',
        'Europe/Tallinn',
        'Europe/Helsinki',
        'Europe/Paris',
        'Europe/Berlin',
        'Europe/Busingen',
        'Europe/Athens',
        'Europe/Budapest',
        'Europe/Dublin',
        'Europe/Rome',
        'Europe/Riga',
        'Europe/Vilnius',
        'Europe/Luxembourg',
        'Europe/Malta',
        'Europe/Amsterdam',
        'Europe/Warsaw',
        'Atlantic/Azores',
        'Atlantic/Madeira',
        'Europe/Lisbon',
        'Europe/Bucharest',
        'Europe/Bratislava',
        'Europe/Ljubljana',
        'Africa/Ceuta',
        'Atlantic/Canary',
        'Europe/Madrid',
        'Europe/Stockholm'
      ];

    var dayjs_script = document.createElement('script');
    var dayjs_tz_script = document.createElement('script');

    dayjs_script.onload = function(e) {
        document.head.appendChild(dayjs_tz_script);
    }
    dayjs_tz_script.onload = function () {
        dayjs.extend(dayjs_plugin_timezone);
        var tz = dayjs.tz.guess();
        if(EU_TIMEZONES.includes(tz)) {
            checkIfAcceptedCookies();
        } else {
            onAcceptedCookies();
        }
    };
    dayjs_script.onerror = function() {
        checkIfAcceptedCookies();
    }
    dayjs_tz_script.onerror = function() {
        checkIfAcceptedCookies();
    }

    dayjs_tz_script.src = 'https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.11.6/plugin/timezone.min.js';
    dayjs_script.src = 'https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.11.6/dayjs.min.js';
    document.head.appendChild(dayjs_script);
}

function checkIfAcceptedCookies() {
    if (!localStorage.acceptedCookies) {
        $('.privacy-banner').show();
        $('.privacy-banner-accept').click(function() {
            $('.privacy-banner').hide()
            onAcceptedCookies();
        });

    } else {
        onAcceptedCookies();
    }
}

function onAcceptedCookies() {
    localStorage.acceptedCookies = 'true';
    initialiseGoogleAnalytics();
    initializeSurvey();
}


function initializeSurvey() {
    $('#survey-link').on('click', function() {
        window.tookSurvey = false;
        window.loadingSurvey = false;
        window.showingSurvey = false;
        showSurvey();
    });

    let now = Date.now() / 1000;

    if(localStorage.ignoredSurveyTimestamp) {
        let timeSinceIgnored = (now - parseFloat(localStorage.ignoredSurveyTimestamp));
        let ignoredSurveyCounter = localStorage.ignoredSurveyCounter || 1;
        let ignoredSurveyTimeout = (ignoredSurveyCounter == 1) ? window.INITIAL_IGNORED_SURVEY_TIMEOUT : window.ADDITIONAL_IGNORED_SURVEY_TIMEOUT;

        if(timeSinceIgnored >= ignoredSurveyTimeout) {
            if(ignoredSurveyCounter == 1) {
                console.log('Reshowing survey since it was initially ignored');
            } else {
                console.log('Reshowing survey since it was ignored again');
            }

            window.tookSurvey = false;
            localStorage.removeItem('surveyUrl');
            localStorage.removeItem('ignoredSurveyTimestamp');
            localStorage.removeItem('surveyCompletedTimestamp');
            localStorage.activeSeconds = window.SHOW_SURVEY_AFTER_SECONDS;
        }
    }

    // If the survey was previously successfully completed
    // then ensure the cache variable exists
    if(window.tookSurvey && !localStorage.ignoredSurveyTimestamp && !localStorage.surveyCompletedTimestamp) {
        localStorage.surveyCompletedTimestamp = now;
    }

    if(localStorage.surveyCompletedTimestamp) {
        let timeSinceCompletedSurvey = (now - parseFloat(localStorage.surveyCompletedTimestamp));
        if(timeSinceCompletedSurvey >= window.RESHOW_SURVEY_TIMEOUT) {
            console.log('Reshowing survey again after it was previously completed');
            window.tookSurvey = false;
            localStorage.removeItem('surveyUrl');
            localStorage.removeItem('ignoredSurveyTimestamp');
            localStorage.removeItem('surveyCompletedTimestamp');
            localStorage.activeSeconds = window.SHOW_SURVEY_AFTER_SECONDS;
        }
    }

    if(!window.tookSurvey) {
        // If the survey was previously taken, but a new survey URL is available
        // then reset the activity counter
        if(localStorage.surveyUrl && localStorage.surveyUrl !== window.SURVEY_URL) {
            localStorage.activeSeconds = 0;
        }
        // Otherwise, if no counter has been previously set
        // then set one now
        else if(!localStorage.activeSeconds) {
            localStorage.activeSeconds = 0;
        }

        localStorage.lastActivityTimestamp = now;

        // Track mouse movement and scrolling
        $(window).scroll(updateSurveyActivity);
        $(document).on('mousemove', updateSurveyActivity);
    }
}


function showSurvey() {
    if(localStorage.acceptedCookies && !window.tookSurvey && !window.loadingSurvey) {
        window.loadingSurvey = true;
        $('#survey-link').text('Loading ...');

        $('#iframe-survey').on('load', function() {
            // This is invoked when the survey is completed
            if(window.showingSurvey) {
                localStorage.removeItem('ignoredSurveyTimestamp');
                localStorage.removeItem('ignoredSurveyCounter');
                localStorage.surveyCompletedTimestamp = Date.now() / 1000;
                $('#survey-link').css('display', 'none');
                closeSurvey();
            } else {
                window.showingSurvey = true;
                $('#dlg-survey').css('display', 'block');
            }
        });
        $('#iframe-survey').attr('src', window.SURVEY_URL);

        // This is invoked when the survey is ignored by clicking the exit button
        $("#dlg-survey-close").on("click", function() {
            closeSurvey();
            localStorage.removeItem('surveyCompletedTimestamp');
            localStorage.ignoredSurveyTimestamp = Date.now() / 1000;
            localStorage.ignoredSurveyCounter = localStorage.ignoredSurveyCounter ? (parseInt(localStorage.ignoredSurveyCounter) + 1) : 1;
        });
    }
}

function closeSurvey() {
    console.info('Took or ignored survey');
    window.tookSurvey = true;
    localStorage.surveyUrl = window.SURVEY_URL;
    $('#dlg-survey').css('display', 'none');
    $('#iframe-survey').off('load');
    $('#iframe-survey').attr('src', '');
    $('#survey-link').text('Feedback');
}

function updateSurveyActivity() {
    let now = Date.now() / 1000;
    let elapsed = now - parseFloat(localStorage.lastActivityTimestamp);

    if(!elapsed || elapsed < 1) {
        return
    }
    localStorage.lastActivityTimestamp = now;

    if(elapsed > 5*60) { // If more than 5min elapsed, then assume the user walked away so ignore this activity
        return;
    }

    let totalSeconds = parseFloat(localStorage.activeSeconds) + elapsed;
    localStorage.activeSeconds = totalSeconds;
    if(totalSeconds >= window.SHOW_SURVEY_AFTER_SECONDS) {
        showSurvey();
    }
}


function addScrollToTopButton() {
    $(window).scroll(function() {
        var footertotop = ($('footer').position().top);
        var scrolltop = $(document).scrollTop() + window.innerHeight;
        var difference = scrolltop-footertotop - 40;

        if (scrolltop > footertotop) {
            $('.go-top').css({'bottom' : difference});
        }else{
            $('.go-top').css({'bottom' : 0});
        };

        if ($(this).scrollTop() > 200) {
            $('.go-top').fadeIn(200);
        } else {
            $('.go-top').fadeOut(200);
        }
    });

    $('.go-top').click(function(event) {
        event.preventDefault();
        $('html, body').animate({scrollTop: 0}, 300);
    })
}
