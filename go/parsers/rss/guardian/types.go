package main

import (
	"encoding/xml"
	"net/url"
)

/*

ITEM EXAMPLE:
<item>
	<title>
		‘They killed him’: widow confronts Peru's president over Covid-19 deaths
	</title>
	<link>
		https://www.theguardian.com/global-development/2020/jul/23/peru-coronavirus-deaths-celia-capira-martin-vizcarra
	</link>
	<description>
		<p>Martín Vizcarra announced emergency decree putting health ministry in charge of system
		after Celia Capira chased his truck</p><p>As the presidential motorcade pulled away
		from the main hospital in Peru’s second city – fleeing an angry protest by medical
		staff and relatives of Covid-19 patients – one woman broke away from the crowd.</p>
		<p>Celia Capira ran sobbing after the truck carrying the president, Martín Vizcarra,
		yelling for him to go and see for himself the grim conditions at the hospital, where
		her husband was fighting for his life.</p>
		<p><span>Related:</span>
		<a href="https://www.theguardian.com/global-development/2020/may/20/peru-coronavirus-lockdown-new-cases">
		Peru’s coronavirus response was ‘right on time’ – so why isn't it working?</a> </p>
		<p> <span>Related: </span>
		<a href="https://www.theguardian.com/global-development/2020/may/07/peru-jungle-iquitos-coronavirus-covid-19">
		‘We are living in a catastrophe’: Peru's jungle capital choking for breath as Covid-19
		hits</a> </p>
		<a href="https://www.theguardian.com/global-development/2020/jul/23/peru-coronavirus-deaths-celia-capira-martin-vizcarra">
		Continue reading...</a>
	</description>
	<category domain="https://www.theguardian.com/global-development/global-development">Global development</category>
	<category domain="https://www.theguardian.com/world/peru">Peru</category>
	<category domain="https://www.theguardian.com/world/coronavirus-outbreak">Coronavirus outbreak</category>
	<category domain="https://www.theguardian.com/world/americas">Americas</category>
	<category domain="https://www.theguardian.com/world/world">World news</category>
	<pubDate>Thu, 23 Jul 2020 09:30:38 GMT</pubDate>
	<guid isPermaLink="false">
		http://www.theguardian.com/global-development/2020/jul/23/peru-coronavirus-deaths-celia-capira-martin-vizcarra
	</guid>
	<media:content width="140" url="https://i.guim.co.uk/img/media/387ebb8dd4cf2cffc3350444ba0bb614962b2b4d/498_336_3736_2243/master/3736.jpg?width=140&quality=85&auto=format&fit=max&s=9b8784b99cd2d3102dc324f514facc81">
		<media:credit scheme="urn:ebu">Photograph: Denis Mayhua</media:credit>
	</media:content>
	<media:content width="460" url="https://i.guim.co.uk/img/media/387ebb8dd4cf2cffc3350444ba0bb614962b2b4d/498_336_3736_2243/master/3736.jpg?width=460&quality=85&auto=format&fit=max&s=bfc6ff1d92f2a073eda9cb6d95fa5151">
		<media:credit scheme="urn:ebu">Photograph: Denis Mayhua</media:credit>
	</media:content>
	<dc:creator>Dan Collyns in Lima</dc:creator>
	<dc:date>2020-07-23T09:30:38Z</dc:date>
</item>

 */

type RSS struct {
	XMLName 		xml.Name	`xml:"rss"`
	Channel			Channel		`xml:"channel"`
}

type Channel struct {
	Item			[]Item		`xml:"item"`
}

type Item struct {
	Title 			string 		`xml:"title"`
	Link			url.URL		`xml:"ink"`
	Description		string		`xml:"description"`
	Category		[]Category	`xml:"category"`
}

type Category struct {
	Key				string		`xml:"domain,attr"`
	Value			string		`xml:",chardata"`
}
